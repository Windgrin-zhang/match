from ultralytics import YOLO
import cv2

# 加载预训练的YOLOv8n模型
model = YOLO('yolov8x.pt')  # 或 yolov8n.yaml 自定义训练模型

# 加载图片
img_path = 'C:\\match\\Amatch-model\\ultralytics-main\\demo\\tang.jpg'  # 替换为你的图片路径
img = cv2.imread(img_path)

# 使用模型进行目标检测（results为一个Results列表）
results = model(img)

# 遍历检测结果
for r in results:
    # 在图像上绘制检测框和标签
    annotated_img = r.plot()

# 显示图像
cv2.imshow('YOLOv8x Detection', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图像
cv2.imwrite('detected_result.jpg', annotated_img)
