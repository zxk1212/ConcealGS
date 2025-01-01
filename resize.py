from PIL import Image

def resize_image(input_path, output_path, size=(64, 64)):
    with Image.open(input_path) as img:
        resized_img = img.resize(size)
        resized_img.save(output_path)

if __name__ == "__main__":
    input_path = '/data/hyliu/Stega_GS_proj/gaussian-splatting/n1/2.png'
    output_path = '/data/hyliu/Stega_GS_proj/gaussian-splatting/n1/2_256.png'
    resize_image(input_path, output_path, size=(256, 256))
    
    input_path = '/data/hyliu/Stega_GS_proj/gaussian-splatting/n1/5.png'
    output_path = '/data/hyliu/Stega_GS_proj/gaussian-splatting/n1/5_256.png'
    resize_image(input_path, output_path, size=(256, 256))
    
    input_path = '/data/hyliu/Stega_GS_proj/gaussian-splatting/n1/6.png'
    output_path = '/data/hyliu/Stega_GS_proj/gaussian-splatting/n1/6_256.png'
    resize_image(input_path, output_path, size=(256, 256))
    
    input_path = '/data/hyliu/Stega_GS_proj/gaussian-splatting/n1/7.png'
    output_path = '/data/hyliu/Stega_GS_proj/gaussian-splatting/n1/7_256.png'
    resize_image(input_path, output_path, size=(256, 256))

# python resize.py