import cv2
import os
import pandas as pd
from ultralytics import YOLO


def resize_if_small(image_path, min_size=640):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        scale_factor = min_size / min(h, w)
        new_size = (int(w * scale_factor), int(h * scale_factor))
        resized_img = cv2.resize(img, new_size)
        return resized_img
    return img


def format_filename(filename):
    """根据要求将文件名中的数值部分补充到四位数"""
    name, ext = os.path.splitext(filename)
    formatted_name = name.zfill(4)
    return formatted_name


def find_image_path(base_path, filename):
    """尝试查找具有多种扩展名的图片文件"""
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        potential_path = os.path.join(base_path, filename + ext)
        if os.path.exists(potential_path):
            return potential_path
    return None


def export_fea_predictions(model_path, test_images_path, output_csv_path, order_csv_path, min_size=640):
    model = YOLO(model_path)
    order_df = pd.read_csv(order_csv_path)

    # 读取顺序和文件名
    filenames_in_order = order_df.iloc[:, 1].tolist()
    id_list = order_df.iloc[:, 0].tolist()

    data = []

    # 按照顺序遍历文件
    for i, filename in enumerate(filenames_in_order):
        formatted_filename = format_filename(filename)
        image_path = find_image_path(test_images_path, formatted_filename)

        if image_path is None:
            print(f"Image {formatted_filename} not found, skipping.")
            data.append([id_list[i], 0, 0, 0, 0])
            continue

        image = resize_if_small(image_path, min_size=min_size)
        results = model.predict(source=image)

        # 初始化预测值
        boundary, calcification, direction, shape = 0, 0, 0, 0

        for result in results:
            for box in result.boxes:
                cls_index = int(box.cls.cpu().numpy()[0])

                # 更新分类标签
                if cls_index == 0:
                    boundary = 0
                elif cls_index == 1:
                    boundary = 1
                elif cls_index == 2:
                    calcification = 0
                elif cls_index == 3:
                    calcification = 1
                elif cls_index == 4:
                    direction = 0
                elif cls_index == 5:
                    direction = 1
                elif cls_index == 6:
                    shape = 0
                elif cls_index == 7:
                    shape = 1

        data.append([id_list[i], boundary, calcification, direction, shape])

    # 创建 DataFrame，并确保列顺序正确
    df = pd.DataFrame(data, columns=['id', 'boundary', 'calcification', 'direction', 'shape'])
    df.to_csv(output_csv_path, index=False)
    print(f"Prediction results saved to {output_csv_path}")


def test_prediction():
    model_path = './runs/detect/train18/weights/best.pt'
    test_images_path = './test_A/A_test_fea/A/images'
    output_csv_path = './runs/csv/fea_pre.csv'
    order_csv_path = './runs/csv/fea_order.csv'

    export_fea_predictions(model_path, test_images_path, output_csv_path, order_csv_path, min_size=640)

    # 检查文件是否正确生成
    assert os.path.exists(output_csv_path), "Output CSV file was not created."


# 运行测试函数
test_prediction()
