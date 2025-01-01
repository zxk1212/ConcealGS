#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# final version train with weight mask;
# motified by piang 2024/08/26
#  this is the py version with the SOTA performance

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch.nn.functional as F


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def load_watermark(wm_path):
    img = cv2.imread(wm_path).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


def training(
    dataset, opt, pipe, 
    testing_iterations, saving_iterations, checkpoint_iterations, 
    checkpoint, debug_from, args
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    gaussians_ori = GaussianModel(dataset.sh_degree)
    # scene_ori = Scene(dataset, gaussians_ori, shuffle=False)
    gaussians_ori.training_setup(opt)
    
    gaussians_stg = GaussianModel(dataset.sh_degree, hidden=True)
    scene_stg = Scene(dataset, gaussians_stg, shuffle=False)
    print("Decoder only:", args.decoder_only)
    gaussians_stg.training_setup(opt, args.decoder_only)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians_stg.restore(model_params, opt)
        gaussians_ori.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    wm_img = load_watermark(args.watermark_path)
    
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians_stg.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians_stg.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene_stg.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        check_cam = scene_stg.getCheckCameras()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg_stg = render(viewpoint_cam, gaussians_stg, pipe, bg)
        
        image_stg, viewspace_point_tensor_stg, visibility_filter_stg, radii_stg = render_pkg_stg["render"], render_pkg_stg["viewspace_points"], render_pkg_stg["visibility_filter"], render_pkg_stg["radii"]
        
        check_pkg_stg = render(check_cam, gaussians_stg, pipe, bg)
        check_pkg_ori = render(check_cam, gaussians_ori, pipe, bg)
        
        check_img_stg = check_pkg_stg["render"]
        check_img_ori = check_pkg_ori["render"]
        gt_check = check_cam.original_image.cuda()
        
        resize_transform = transforms.Resize((args.wm_size, args.wm_size))
        image_stg_resized = resize_transform(check_img_stg)
        image_ori_resized = resize_transform(check_img_ori)
        input_image = torch.stack([image_ori_resized, image_stg_resized], dim=0)
        
        output = gaussians_stg.detector(input_image)
        output_ori = output[0]
        output_stg = output[1]
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image_stg, gt_image)
        
        # loss for watermark
        loss_watermark = l1_loss(output_stg, wm_img.to(output_stg.device))
        loss_target = l1_loss(output_ori, image_ori_resized)
        loss_check = l1_loss(check_img_stg, gt_check)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_stg, gt_image)) + loss_watermark + loss_target * args.target_weight + loss_check 
        # + (1.0 - ssim(output_stg, wm_img.to(output_stg.device))) * 0.01
        wm_psnr = psnr(output_stg, wm_img.to(output_stg.device))
        
        # loss.backward()
        iter_end.record()
        
        param_list = [gaussians_stg._features_dc, gaussians_stg._features_rest]
        param_name = ["_features_dc", "_features_rest"]
        
        gaussians_stg.optimizer.zero_grad()  
        loss_watermark.backward(retain_graph=True)
        param_dict1 = {}
        for name1, parameter in gaussians_stg.detector.named_parameters():
            if parameter.grad is None:
                continue
            parameter_grad = parameter.grad.clone().detach()
            param_dict1[name1] = parameter_grad

        gaussians_stg.optimizer.zero_grad()  
        loss_target.backward(retain_graph=True)
        param_dict2 = {}
        for name2, parameter in gaussians_stg.detector.named_parameters():
            if parameter.grad is None:
                continue
            parameter_grad = parameter.grad.clone().detach()
            param_dict2[name2] = parameter_grad
        
        param_dict_final = {}
        for item in param_dict2.keys():
            cos = F.cosine_similarity(param_dict1[item].view(1, -1), param_dict2[item].view(1, -1))
            param_dict_final[item] = torch.ones_like(param_dict1[item]) * (1 / (1 + torch.exp(-cos)))     
        
        
        gaussians_stg.optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                testing_iterations, scene_stg, render, (pipe, background),
                loss_watermark, loss_target, loss_check, wm_psnr
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene_stg.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians_stg.max_radii2D[visibility_filter_stg] = torch.max(gaussians_stg.max_radii2D[visibility_filter_stg], radii_stg[visibility_filter_stg])
                gaussians_stg.add_densification_stats(viewspace_point_tensor_stg, visibility_filter_stg)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians_stg.densify_and_prune(opt.densify_grad_threshold, 0.005, scene_stg.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians_stg.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                for name, parameter in gaussians_stg.detector.named_parameters():
                    if name not in param_dict_final.keys():
                        continue
                    mask = param_dict_final[name]
                    parameter.grad = parameter.grad * mask
                    
                gaussians_stg.optimizer.step()
                gaussians_stg.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians_stg.capture(), iteration), scene_stg.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer, iteration, Ll1, loss, l1_loss, elapsed, 
    testing_iterations, scene : Scene, renderFunc, renderArgs,
    loss_watermark, loss_target, loss_check, wm_psnr
):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/loss_watermark', loss_watermark.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/loss_target', loss_target.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/loss_check', loss_check.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/wm_psnr', wm_psnr.mean().item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[int(i * 5000) for i in range(1, 21)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[int(i * 5000) for i in range(1, 21)])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default="")
    parser.add_argument("--target_weight", type=float, default=1e-2)
    parser.add_argument("--watermark_path", type=str, default="")
    parser.add_argument("--wm_size", type=int, default="64")
    parser.add_argument("--decoder_only", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args), op.extract(args), pp.extract(args), 
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
        args.start_checkpoint, args.debug_from, args
    )

    # All done
    print("\nTraining complete.")
