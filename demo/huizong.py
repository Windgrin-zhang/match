import random
import os
import shutil


def copy_images_and_first_line_of_txt_to_directories(train_cla_source_directory, Real_train_image_target_directory, Real_train_txt_target_directory):
    # 如果目标目录不存在，则创建它们
    if not os.path.exists(Real_train_image_target_directory):
        os.makedirs(Real_train_image_target_directory)

    if not os.path.exists(Real_train_txt_target_directory):
        os.makedirs(Real_train_txt_target_directory)

    # 使用 os.walk 遍历源目录及其子目录中的所有文件
    for root, dirs, files in os.walk(train_cla_source_directory):
        for filename in files:
            # 处理图片文件
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):  # 根据需要添加更多图片格式
                # 构建源文件路径
                source_file = os.path.join(root, filename)

                # 构建目标文件路径
                target_file = os.path.join(Real_train_image_target_directory, filename)

                # 如果目标目录中已存在同名文件，进行重命名
                if os.path.exists(target_file):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(target_file):
                        target_file = os.path.join(Real_train_image_target_directory, f"{base}_{counter}{ext}")
                        counter += 1

                # 复制图片文件
                shutil.copy(source_file, target_file)
                print(f"Copied image: {source_file} to {target_file}")

            # 处理.txt文件，只复制第一行
            elif filename.endswith('.txt'):
                # 构建源文件路径
                source_file = os.path.join(root, filename)

                # 构建目标文件路径
                target_file = os.path.join(Real_train_txt_target_directory, filename)

                # 如果目标目录中已存在同名文件，进行重命名
                if os.path.exists(target_file):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(target_file):
                        target_file = os.path.join(Real_train_txt_target_directory, f"{base}_{counter}{ext}")
                        counter += 1

                # 读取源文件的第一行并写入目标文件
                with open(source_file, 'r', encoding='utf-8') as src:
                    first_line = src.readline()  # 读取第一行
                    if first_line:  # 检查是否读取到内容
                        with open(target_file, 'w', encoding='utf-8') as dst:
                            dst.write(first_line)  # 写入目标文件
                            print(f"Copied first line from: {source_file} to {target_file}")


# # 指定源目录和目标目录（相对路径或绝对路径）
# train_cla_source_dir = './train_cla/train'
# Real_train_image_target_dir = './train_cla/Real_train/images'
# Real_train_txt_target_dir = './train_cla/Real_train/labels'

#copy_images_and_first_line_of_txt_to_directories(train_cla_source_dir, Real_train_image_target_dir, Real_train_txt_target_dir)




# #  取出十分之一作为验证集
def copy_random_images_with_labels(val_source_image_dir, val_source_label_dir, val_target_image_dir, val_target_label_dir, fraction=0.1):
    # 如果目标图片和标签目录不存在，则创建它们
    if not os.path.exists(val_target_image_dir):
        os.makedirs(val_target_image_dir)
    if not os.path.exists(val_target_label_dir):
        os.makedirs(val_target_label_dir)

    # 存储所有图片文件的路径
    all_images = []

    # 遍历源图片目录及其子目录中的所有图片文件
    for root, dirs, files in os.walk(val_source_image_dir):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):  # 根据需要添加更多图片格式
                all_images.append(os.path.join(root, filename))

    # 计算要选出的图片数量
    total_images = len(all_images)
    num_images_to_copy = max(1, int(total_images * fraction))  # 至少选择1个文件

    # 随机选择指定比例的图片
    selected_images = random.sample(all_images, num_images_to_copy)

    # 复制随机选择的图片及其对应的标签到目标目录
    for source_image in selected_images:
        # 构建目标图片文件路径
        relative_image_path = os.path.relpath(source_image, val_source_image_dir)
        target_image_file = os.path.join(val_target_image_dir, relative_image_path)
        target_image_subdir = os.path.dirname(target_image_file)  # 获取目标图片的子目录

        # 创建目标图片子目录（如果不存在）
        if not os.path.exists(target_image_subdir):
            os.makedirs(target_image_subdir)

        # 复制图片文件
        shutil.copy(source_image, target_image_file)
        print(f"Copied image: {source_image} to {target_image_file}")

        # 找到对应的标签文件
        image_filename = os.path.basename(source_image)
        label_filename = os.path.splitext(image_filename)[0] + '.txt'  # 假设标签文件为 .txt
        source_label = os.path.join(val_source_label_dir, label_filename)

        if os.path.exists(source_label):  # 如果标签文件存在
            # 构建目标标签文件路径
            relative_label_path = os.path.relpath(source_label, val_source_label_dir)
            target_label_file = os.path.join(val_target_label_dir, relative_label_path)
            target_label_subdir = os.path.dirname(target_label_file)  # 获取目标标签的子目录

            # 创建目标标签子目录（如果不存在）
            if not os.path.exists(target_label_subdir):
                os.makedirs(target_label_subdir)

            # 复制标签文件
            shutil.copy(source_label, target_label_file)
            print(f"Copied label: {source_label} to {target_label_file}")
        else:
            print(f"No label found for image: {source_image}")

# # 指定源图片和标签目录以及目标图片和标签目录
# val_source_image_dir = './train_cla/Real_train/images'
# val_source_label_dir = './train_cla/Real_train/labels'
# val_target_image_dir = './train_cla/val/images'
# val_target_label_dir = './train_cla/val/labels'

# 调用函数，选择并复制十分之一的图片和标签
#copy_random_images_with_labels(val_source_image_dir, val_source_label_dir, val_target_image_dir, val_target_label_dir, fraction=0.1)




