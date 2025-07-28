import cv2
import os
import pandas as pd
from ultralytics import YOLO


def resize_if_small(image_path, min_size=640):
    # 使用 OpenCV 读取图片
    img = cv2.imread(image_path)
    h, w = img.shape[:2]  # 获取图片的高度和宽度

    # 如果宽度或高度小于设定的最小尺寸，则进行调整
    if h < min_size or w < min_size:
        # 按照保持比例的方式缩放图片，使其最小边等于 min_size
        scale_factor = min_size / min(h, w)
        new_size = (int(w * scale_factor), int(h * scale_factor))
        resized_img = cv2.resize(img, new_size)
        return resized_img
    else:
        # 如果尺寸已经符合要求，返回原始图片
        return img


def format_filename(filename):
    """根据要求将文件名中的数值部分补充到四位数"""
    name, ext = os.path.splitext(filename)
    # 提取数值部分并补充前导零
    formatted_name = name.zfill(4)
    return formatted_name


def find_image_path(base_path, filename):
    """尝试查找具有多种扩展名的图片文件"""
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:  # 常见图片格式
        potential_path = os.path.join(base_path, filename + ext)
        if os.path.exists(potential_path):
            return potential_path
    return None


def export_cla_predictions(model_path, test_images_path, output_csv_path, order_csv_path, min_size=640):
    # 检查 cla_pre.csv 文件是否存在，如果不存在则创建一个空的文件
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, 'w') as f:
            pass  # 创建空文件

    # 加载 YOLO 模型
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 读取 order.csv 文件并获取文件名和对应的序号
    order_df = pd.read_csv(order_csv_path)
    print("First 5 rows of order.csv:", order_df.head())

    filenames_in_order = order_df.iloc[:, 1].tolist()
    id_list = order_df.iloc[:, 0].tolist()

    data = []

    for i, filename in enumerate(filenames_in_order):
        formatted_filename = format_filename(filename)
        image_path = find_image_path(test_images_path, formatted_filename)

        if image_path is None:
            print(f"Image {formatted_filename} not found, skipping.")
            # 即使未找到图像，仍然记录在 CSV 中，并设定 0 作为默认标签
            data.append([id_list[i], 0])
            continue
        else:
            print(f"Image found: {image_path}")

        image = resize_if_small(image_path, min_size=min_size)

        # 使用 YOLO 模型进行预测
        try:
            results = model.predict(source=image_path)
            print(f"Prediction results for {formatted_filename}: {results}")
        except Exception as e:
            print(f"Error during prediction for {formatted_filename}: {e}")
            # 出现错误时，也将其添加到 CSV，并设定 0 作为默认标签
            data.append([id_list[i], 0])
            continue

        # 检查是否有预测结果
        if not results or len(results[0].boxes) == 0:
            print(f"No prediction results for {formatted_filename}.")
            # 没有检测结果时，将其添加到 CSV，分类标签设为 0
            data.append([id_list[i], 0])
            continue

        # 处理检测到的结果
        predicted_classes = set()
        for result in results:
            for box in result.boxes:
                predicted_class = int(box.cls[0])
                predicted_classes.add(predicted_class)

        print(f"Predicted classes for {formatted_filename}: {predicted_classes}")

        image_id = id_list[i]
        # 如果有检测结果，保存第一个检测到的类别；否则默认标签为 0
        if predicted_classes:
            data.append([image_id, list(predicted_classes)[0]])
        else:
            data.append([image_id, 0])  # 没有检测到结果时设为 0

    # 保存预测结果到 CSV
    df = pd.DataFrame(data, columns=['image_id', 'predicted_class'])
    df.to_csv(output_csv_path, index=False)
    print(f"Prediction results saved to {output_csv_path}")




def test_prediction():
    model_path = './runs/detect/train21/weights/best.pt'
    test_images_path = './test_A/A_test_cla/Real_A/images'
    output_csv_path = './runs/csv/cla_pre.csv'
    order_csv_path = './runs/csv/cla_order.csv'

    # 调用你的函数
    export_cla_predictions(model_path, test_images_path, output_csv_path, order_csv_path, min_size=640)

    # 你可以在这里添加断言，来验证结果是否正确
    assert os.path.exists(output_csv_path), "Output CSV file was not created."