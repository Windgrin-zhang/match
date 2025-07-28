import os
import shutil


def copy_images_and_first_line_of_txt_to_directories_A(source_directory, image_target_directory, txt_target_directory):
    # 如果目标目录不存在，则创建它们
    if not os.path.exists(image_target_directory):
        os.makedirs(image_target_directory)

    if not os.path.exists(txt_target_directory):
        os.makedirs(txt_target_directory)

    # 使用 os.walk 遍历源目录及其子目录中的所有文件
    for root, dirs, files in os.walk(source_directory):
        for filename in files:
            # 处理图片文件
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):  # 根据需要添加更多图片格式
                # 构建源文件路径
                source_file = os.path.join(root, filename)

                # 构建目标文件路径
                target_file = os.path.join(image_target_directory, filename)

                # 如果目标目录中已存在同名文件，进行重命名
                if os.path.exists(target_file):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(target_file):
                        target_file = os.path.join(image_target_directory, f"{base}_{counter}{ext}")
                        counter += 1

                # 复制图片文件
                shutil.copy(source_file, target_file)
                print(f"Copied image: {source_file} to {target_file}")

            # 处理.txt文件，只复制第一行
            elif filename.endswith('.txt'):
                # 构建源文件路径
                source_file = os.path.join(root, filename)

                # 构建目标文件路径
                target_file = os.path.join(txt_target_directory, filename)

                # 如果目标目录中已存在同名文件，进行重命名
                if os.path.exists(target_file):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(target_file):
                        target_file = os.path.join(txt_target_directory, f"{base}_{counter}{ext}")
                        counter += 1

                # 读取源文件的第一行并写入目标文件
                with open(source_file, 'r', encoding='utf-8') as src:
                    first_line = src.readline()  # 读取第一行
                    if first_line:  # 检查是否读取到内容
                        with open(target_file, 'w', encoding='utf-8') as dst:
                            dst.write(first_line)  # 写入目标文件
                            print(f"Copied first line from: {source_file} to {target_file}")


# # 指定源目录和目标目录（相对路径或绝对路径）
# source_dir = '.demo/test_A/A_test_cla/A'
# image_target_dir = '.demo/test_A/A_test_cla/Real_A/images'
# txt_target_dir = '.demo/test_A/A_test_cla/Real_A/labels'
#
# copy_images_and_first_line_of_txt_to_directories_A(source_dir, image_target_dir, txt_target_dir)