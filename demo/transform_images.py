import os
import struct
import gzip
import numpy as np
from PIL import Image
import shutil

def process_image(image_path):
    """处理图片：转换为灰度并调整大小"""
    img = Image.open(image_path).convert('L')  # 灰度
    img = img.resize((28, 28))                 # 缩放为28×28
    img = np.asarray(img, dtype=np.uint8)
    return img

def images_to_idx3_ubyte(image_folder, output_file):
    """将图片文件夹转换为IDX3格式"""
    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    if not image_files:
        print(f"警告：在 {image_folder} 中没有找到图片文件")
        return False

    num_images = len(image_files)
    rows, cols = 28, 28

    # Header
    header = struct.pack('>IIII', 2051, num_images, rows, cols)

    # Image data
    image_data = bytearray()
    for img_file in image_files:
        img = process_image(img_file)
        image_data.extend(img.flatten())

    # Combine
    with open(output_file, 'wb') as f:
        f.write(header + image_data)

    print(f"保存了 {num_images} 张图片到 {output_file}")
    return True

def compress_to_gz(input_file, output_gz_file):
    """压缩文件为gz格式"""
    if os.path.exists(input_file):
        with open(input_file, 'rb') as f_in:
            with gzip.open(output_gz_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"压缩完成：{output_gz_file}")
        # 删除临时文件
        os.remove(input_file)
    else:
        print(f"错误：文件 {input_file} 不存在")

def process_folder(folder_path):
    """处理单个文件夹，只生成图片的gz文件"""
    folder_name = os.path.basename(folder_path)
    parent_dir = os.path.dirname(folder_path)
    
    print(f"\n处理文件夹: {folder_path}")
    
    # 查找图片文件夹
    image_folders = []
    
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            if dir_name.lower() in ['images', 'image', 'img']:
                image_folders.append(os.path.join(root, dir_name))
    
    # 如果没有找到标准的图片文件夹，检查当前文件夹
    if not image_folders:
        # 检查当前文件夹是否包含图片
        files = os.listdir(folder_path)
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) for f in files):
            image_folders.append(folder_path)
    
    # 处理图片
    for i, img_folder in enumerate(image_folders):
        suffix = f"_{i+1}" if len(image_folders) > 1 else ""
        temp_img_file = os.path.join(parent_dir, f"{folder_name}_images{suffix}.idx3-ubyte")
        gz_img_file = os.path.join(parent_dir, f"{folder_name}_images{suffix}.idx3-ubyte.gz")
        
        if images_to_idx3_ubyte(img_folder, temp_img_file):
            compress_to_gz(temp_img_file, gz_img_file)

def main():
    """主函数：只处理test_A文件夹中的A文件夹"""
    # 只处理test_A文件夹中的A文件夹（不包括Real_A）
    folders_to_process = [
        "demo/test_A/A_test_cla/A",  # A_test_cla中的A文件夹
        "demo/test_A/A_test_fea/A"   # A_test_fea中的A文件夹
    ]
    
    for folder in folders_to_process:
        if os.path.exists(folder):
            process_folder(folder)
        else:
            print(f"警告：文件夹 {folder} 不存在")

if __name__ == "__main__":
    main()
