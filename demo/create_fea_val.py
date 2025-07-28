import os
import shutil
import random
from pathlib import Path


def create_val_folder(data_dir):
    # 定义 val 和 labels 文件夹路径
    val_dir = Path(data_dir) / 'val'
    labels_dir = Path(data_dir) / 'train' / 'labels'

    # 创建 val 文件夹及其子文件夹
    val_images_path = val_dir / 'images'
    val_labels_path = val_dir / 'labels'

    val_images_path.mkdir(parents=True, exist_ok=True)  # 创建 val/images 子文件夹
    val_labels_path.mkdir(parents=True, exist_ok=True)  # 创建 val/labels 子文件夹
    labels_dir.mkdir(parents=True, exist_ok=True)  # 确保 labels 文件夹存在

    # 获取训练集中所有图像文件
    train_images_dir = Path(data_dir) / 'train' / 'images'
    all_images = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not all_images:
        print(f"警告: {train_images_dir} 中没有找到图像文件。")
        return

    # 随机选择 1/10 的图像（至少选择一个）
    num_val_images = max(1, len(all_images) // 10)

    # 如果可用的图像少于选择的数量，则调整选择数量
    if num_val_images > len(all_images):
        num_val_images = len(all_images)

    selected_images = random.sample(all_images, num_val_images)

    # 合并所有图像的标签并保存到 labels
    for img_file in selected_images:
        combined_labels = []

        for feature in ['boundary_labels', 'calcification_labels', 'direction_labels', 'shape_labels']:
            train_labels_dir = Path(data_dir) / 'train' / feature
            label_file = img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            label_file_path = train_labels_dir / label_file

            # 如果标签文件存在，读取并添加到 combined_labels 中
            if os.path.exists(label_file_path):
                print(f"读取标签文件: {label_file_path}")
                with open(label_file_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) == 0:
                        print(f"警告: 标签文件为空，跳过: {label_file_path}")
                        continue  # 跳过空文件
                    elif len(lines) == 1:
                        # 只有一行的情况，直接按规则录入
                        label_value = int(lines[0].split()[0].strip())  # 标签值是0或1
                        position_info = " ".join(lines[0].split()[1:])  # 获取高度、宽度等位置信息
                    else:
                        # 多行的情况，只录入第一行
                        label_value = int(lines[0].split()[0].strip())  # 标签值是0或1
                        position_info = " ".join(lines[0].split()[1:])  # 获取高度、宽度等位置信息

                    # 根据不同类别的标签值进行转换
                    if feature == 'boundary_labels':
                        new_label_value = label_value  # 0 或 1
                    elif feature == 'calcification_labels':
                        new_label_value = label_value + 2  # 2 或 3
                    elif feature == 'direction_labels':
                        new_label_value = label_value + 4  # 4 或 5
                    elif feature == 'shape_labels':
                        new_label_value = label_value + 6  # 6 或 7

                    # 构建合并后的标签行
                    merged_label = f"{new_label_value} {position_info}"
                    combined_labels.append(merged_label)
            else:
                print(f"警告: 标签文件不存在: {label_file_path}")

        # 将合并后的标签写入新的文件
        if combined_labels:
            merged_label_file_path = val_labels_path / label_file
            with open(merged_label_file_path, 'w') as f:
                f.write("\n".join(combined_labels))

            # 复制图像文件到 val/images 文件夹
            src_img_path = train_images_dir / img_file
            shutil.copy(src_img_path, val_images_path / img_file)

    print("验证集文件夹创建完成，已随机选择图像和合并标签。")

    # 整合所有标签到 train/labels 目录
    for img_file in all_images:
        combined_labels = []

        for feature in ['boundary_labels', 'calcification_labels', 'direction_labels', 'shape_labels']:
            train_labels_dir = Path(data_dir) / 'train' / feature
            label_file = img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            label_file_path = train_labels_dir / label_file

            # 如果标签文件存在，读取并添加到 combined_labels 中
            if os.path.exists(label_file_path):
                print(f"读取标签文件: {label_file_path}")
                with open(label_file_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) == 0:
                        print(f"警告: 标签文件为空，跳过: {label_file_path}")
                        continue  # 跳过空文件
                    elif len(lines) == 1:
                        # 只有一行的情况，直接按规则录入
                        label_value = int(lines[0].split()[0].strip())  # 标签值是0或1
                        position_info = " ".join(lines[0].split()[1:])  # 获取高度、宽度等位置信息
                    else:
                        # 多行的情况，只录入第一行
                        label_value = int(lines[0].split()[0].strip())  # 标签值是0或1
                        position_info = " ".join(lines[0].split()[1:])  # 获取高度、宽度等位置信息

                    # 根据不同类别的标签值进行转换
                    if feature == 'boundary_labels':
                        new_label_value = label_value  # 0 或 1
                    elif feature == 'calcification_labels':
                        new_label_value = label_value + 2  # 2 或 3
                    elif feature == 'direction_labels':
                        new_label_value = label_value + 4  # 4 或 5
                    elif feature == 'shape_labels':
                        new_label_value = label_value + 6  # 6 或 7

                    # 构建合并后的标签行
                    merged_label = f"{new_label_value} {position_info}"
                    combined_labels.append(merged_label)
            else:
                print(f"警告: 标签文件不存在: {label_file_path}")

        # 将合并后的标签写入 train/labels
        if combined_labels:
            merged_label_file_path = labels_dir / label_file
            with open(merged_label_file_path, 'w') as f:
                f.write("\n".join(combined_labels))

    print("所有标签已整合到 train/labels 目录。")


# 使用示例
data_directory = ("./train_fea")
create_val_folder(data_directory)