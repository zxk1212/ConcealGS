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

import torch
from scene import Scene
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import torchvision.transforms as transforms
import cv2
import numpy as np
from time import time


def psnr(img1, img2):
    mse_mask = ((img1 - img2) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return 20* torch.log10(1.0 / torch.sqrt(mse_mask))


def load_watermark(wm_path):
    img = cv2.imread(wm_path).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scene, watermark_path, wm_size):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    check_cam = scene.getCheckCameras()
    gt_check = check_cam.original_image[0:3, :, :]
    with torch.no_grad():
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            # continue
            rendering = render(view, gaussians, pipeline, background)["render"]
            
            pic_ori = rendering
            resize_transform = transforms.Resize((64, 64))
            image_stg_resized = resize_transform(pic_ori)
            wm_img = load_watermark(watermark_path)
            output = gaussians.detector(image_stg_resized.unsqueeze(0))
            psnr1 = psnr(output, image_stg_resized.unsqueeze(0).to(output.device))
            print("ori psnr:", psnr1)
            
            # torchvision.utils.save_image(output, os.path.join(render_path, '{0:05d}'.format(idx) + "_ori.png"))
            wm_data1 = output.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
            wm_data1 = (wm_data1 * 255).astype(np.uint8).clip(0, 255)
            # cv2.imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + "_ori.png"), wm_data1)
            
            output = gaussians.detector(image_stg_resized.unsqueeze(0))
            # output = torch.flip(output, dims=[1])
            # torchvision.utils.save_image(output, os.path.join(render_path, '{0:05d}'.format(idx) + "_ori1.png"))
            # torchvision.utils.save_image(image_stg_resized, os.path.join(render_path, '{0:05d}'.format(idx) + "_ori2.png"))
            
            gt = view.original_image[0:3, :, :]
            
            psnr3 = psnr(rendering.unsqueeze(0), gt.unsqueeze(0).to(output.device))
            print("normal psnr:", psnr3)
            
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            
        check_pkg_stg = render(check_cam, gaussians, pipeline, background)
        check_img_stg = check_pkg_stg["render"]
        
        psnr2 = psnr(check_img_stg, gt_check.unsqueeze(0).to(output.device))
        print("checkpic psnr:", psnr2)
        
        resize_transform = transforms.Resize((wm_size, wm_size))
        image_stg_resized = resize_transform(check_img_stg)
        wm_img = load_watermark(watermark_path)
        output = gaussians.detector(image_stg_resized.unsqueeze(0))
        
        psnr1 = psnr(output.unsqueeze(0), wm_img.unsqueeze(0).to(output.device))
        print("watermark psnr:", psnr1)
        
        output = torch.flip(output, dims=[1])
        torchvision.utils.save_image(output, os.path.join(gts_path, "watermark.png"))
        # image_stg_resized1 = image_stg_resized.permute(1, 2, 0).detach().cpu().numpy() * 255
        # cv2.imwrite(os.path.join(gts_path, "watermark.png"), image_stg_resized)
        
        wm_data1 = output.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        # wm_data1 = (wm_data1 - wm_data1.min()) / (wm_data1.max() - wm_data1.min())
        wm_data1 = (wm_data1 * 255).astype(np.uint8).clip(0, 255)
        # wm_data1 = cv2.cvtColor(wm_data1, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(gts_path, "watermark1.png"), wm_data1)
        
        # wm_data2 = check_img_stg.unsqueeze(0).permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        # wm_data2 = (wm_data2 * 255).astype(np.uint8)
        # cv2.imwrite(os.path.join(gts_path, "check_view1.png"), wm_data2)
        
        torchvision.utils.save_image(check_img_stg, os.path.join(gts_path, "check_view1.png"))
        torchvision.utils.save_image(gt_check, os.path.join(gts_path, "check_view_gt.png"))
    
    render_test = False
    # render_test = True
    if render_test:
        test_times = 50
        for i in range(test_times):
            for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
                if idx == 0 and i == 0:
                    time1 = time()
               
                rendering = render(view, gaussians, pipeline, background)["render"]
            
                pic_ori = rendering
                resize_transform = transforms.Resize((64, 64))
                image_stg_resized = resize_transform(pic_ori)
                wm_img = load_watermark(watermark_path)
                output = gaussians.detector(image_stg_resized.unsqueeze(0))
                # print("ori psnr:", psnr1)
                
        time2=time()
        print("FPS:",(len(views)-1)*test_times/(time2-time1))
        
    
    
    

def render_sets(
    dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, 
    watermark_path: str, wm_size: int
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hidden=True)
        # scene_ori = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, hidden=True)
        scene_stg = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene_stg.loaded_iter, scene_stg.getTrainCameras(), gaussians, pipeline, background, scene_stg, watermark_path, wm_size)

        if not skip_test:
            render_set(dataset.model_path, "test", scene_stg.loaded_iter, scene_stg.getTestCameras(), gaussians, pipeline, background, scene_stg, watermark_path, wm_size)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--watermark_path", default="", type=str)
    parser.add_argument("--wm_size", type=int, default="64")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, 
        args.watermark_path, args.wm_size
    )


