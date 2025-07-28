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


def export_cla_predictions(model_path, test_images_path, output_csv_path, min_size=640):
    # 检查 cla_pre.csv 文件是否存在，如果不存在则创建一个空的文件
    if not os.path.exists(output_csv_path):
        # 创建一个空的 DataFrame，指定列名
        empty_df = pd.DataFrame(columns=['id', 'label'])
        # 将空的 DataFrame 写入 CSV 文件
        empty_df.to_csv(output_csv_path, index=False)
        print(f"Created an empty file: {output_csv_path}")

    # 加载 YOLO 模型
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 获取文件夹中的所有图片文件（支持多种扩展名）
    all_images = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    all_images.sort()  # 按文件名顺序排序

    data = []

    for i, filename in enumerate(all_images):
        image_path = os.path.join(test_images_path, filename)

        print(f"Processing image: {image_path}")

        image = resize_if_small(image_path, min_size=min_size)

        # 使用 YOLO 模型进行预测
        try:
            results = model.predict(source=image_path)
            print(f"Prediction results for {filename}: {results}")
        except Exception as e:
            print(f"Error during prediction for {filename}: {e}")
            # 出现错误时，设定 1 作为默认标签
            data.append([i + 1, 1])  # 使用顺序编号作为 ID
            continue

        # 检查是否有预测结果
        if not results or len(results[0].boxes) == 0:
            print(f"No prediction results for {filename}.")
            # 没有检测结果时，分类标签设为 1
            data.append([i + 1, 1])
            continue

        # 处理检测到的结果
        predicted_classes = set()
        for result in results:
            for box in result.boxes:
                predicted_class = int(box.cls[0]) + 1  # 将类别索引加 1 使范围变为 1-6
                predicted_classes.add(predicted_class)

        print(f"Predicted classes for {filename}: {predicted_classes}")

        # 保存第一个检测到的类别；否则默认标签为 1
        if predicted_classes:
            data.append([i + 1, list(predicted_classes)[0]])  # 使用顺序编号作为 ID
        else:
            data.append([i + 1, 1])  # 没有检测到结果时设为 1

    # 保存预测结果到 CSV
    df = pd.DataFrame(data, columns=['id', 'label'])
    df.to_csv(output_csv_path, index=False)
    print(f"Prediction results saved to {output_csv_path}")


# 使用实例
if __name__ == "__main__":
    # 模型文件路径
    model_path = './runs/detect/train21/weights/best.pt'
    # 测试图片目录

    # test_images_path = '../testB/cla'
    # 输出 CSV 文件路径
    output_csv_path = './runs/csv/cla_pre.csv'

    # 调用函数进行预测，最小图片尺寸设为 640
    export_cla_predictions(model_path, test_images_path, output_csv_path, min_size=640)


#
#
# import cv2
# import os
# import pandas as pd
# from ultralytics import YOLO
# import logging
#
# def resize_if_small(image_path, min_size=640):
#     # 使用 OpenCV 读取图片
#     img = cv2.imread(image_path)
#     h, w = img.shape[:2]  # 获取图片的高度和宽度
#
#     # 如果宽度或高度小于设定的最小尺寸，则进行调整
#     if h < min_size or w < min_size:
#         # 按照保持比例的方式缩放图片，使其最小边等于 min_size
#         scale_factor = min_size / min(h, w)
#         new_size = (int(w * scale_factor), int(h * scale_factor))
#         resized_img = cv2.resize(img, new_size)
#         return resized_img
#     else:
#         # 如果尺寸已经符合要求，返回原始图片
#         return img
#
#
# def export_cla_predictions(model_path, test_images_path, output_csv_path, min_size=640):
#     # 检查 cla_pre.csv 文件是否存在，如果不存在则创建一个空的文件
#     if not os.path.exists(output_csv_path):
#         # 创建一个空的 DataFrame，指定列名
#         empty_df = pd.DataFrame(columns=['id', 'label'])
#         # 将空的 DataFrame 写入 CSV 文件
#         empty_df.to_csv(output_csv_path, index=False)
#
#     # 设置日志级别为 ERROR，仅显示错误信息，禁用其他日志输出
#     logging.getLogger("ultralytics").setLevel(logging.ERROR)
#
#     try:
#         model = YOLO(model_path)
#     except Exception as e:
#         return
#
#     # 获取文件夹中的所有图片文件（支持多种扩展名）
#     all_images = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
#     all_images.sort()  # 按文件名顺序排序
#
#     data = []
#
#     for i, filename in enumerate(all_images):
#         image_path = os.path.join(test_images_path, filename)
#
#         image = resize_if_small(image_path, min_size=min_size)
#
#         # 使用 YOLO 模型进行预测
#         try:
#             results = model.predict(source=image_path)
#         except Exception as e:
#             # 出现错误时，设定 1 作为默认标签
#             data.append([i + 1, 1])  # 使用顺序编号作为 ID
#             continue
#
#         # 检查是否有预测结果
#         if not results or len(results[0].boxes) == 0:
#             # 没有检测结果时，分类标签设为 1
#             data.append([i + 1, 1])
#             continue
#
#         # 处理检测到的结果
#         predicted_classes = set()
#         for result in results:
#             for box in result.boxes:
#                 predicted_class = int(box.cls[0]) + 1  # 将类别索引加 1 使范围变为 1-6
#                 predicted_classes.add(predicted_class)
#
#         # 保存第一个检测到的类别；否则默认标签为 1
#         if predicted_classes:
#             data.append([i + 1, list(predicted_classes)[0]])  # 使用顺序编号作为 ID
#         else:
#             data.append([i + 1, 1])  # 没有检测到结果时设为 1
#
#     # 保存预测结果到 CSV
#     df = pd.DataFrame(data, columns=['id', 'label'])
#     df.to_csv(output_csv_path, index=False)
#
# # 使用实例
# if __name__ == "__main__":
#     # 模型文件路径
#     model_path = './runs/detect/train21/weights/best.pt'
#     # 测试图片目录
#     test_images_path = '../testB/cla'
#     # 输出 CSV 文件路径
#     output_csv_path = './runs/csv/cla_pre.csv'
#
#     # 调用函数进行预测，最小图片尺寸设为 640
#     export_cla_predictions(model_path, test_images_path, output_csv_path, min_size=640)
