import os
import csv

# 设置路径
base_path = './A_test_fea/A'  # 替换为您的实际路径
labels_folders = ['boundary_labels', 'calcification_labels', 'direction_labels', 'shape_labels']
fea_order_path = '../runs/csv/fea_order.csv'  # 替换为 fea_order.csv 的实际路径
output_csv_path = '../runs/csv/fea_gt.csv'  # 输出文件名

# 读取 fea_order CSV
with open(fea_order_path, 'r') as fea_order_file:
    fea_order_reader = csv.reader(fea_order_file)
    # 跳过表头（如果有）
    next(fea_order_reader, None)

    # 初始化结果列表
    results = []
    for row in fea_order_reader:
        id_value = row[0]  # 第一列是 ID
        image_name = row[1]  # 第二列是图片名字

        # 创建一个结果行
        result_row = [id_value]

        # 遍历每个标签文件夹
        for folder in labels_folders:
            label_file_path = os.path.join(base_path, folder, f'{image_name}.txt')
            if os.path.isfile(label_file_path):
                with open(label_file_path, 'r') as label_file:
                    first_value = label_file.readline().strip().split()[0]  # 获取第一个数字
                    result_row.append(first_value)
            else:
                result_row.append(0)  # 如果文件不存在，默认值为 0

        # 将结果行添加到结果列表
        results.append(result_row)

# 将结果写入 fea_gt.csv
with open(output_csv_path, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)
    # 写入表头
    csv_writer.writerow(['id', 'boundary', 'calcification', 'direction', 'shape'])
    # 写入结果
    csv_writer.writerows(results)