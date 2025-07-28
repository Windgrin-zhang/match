
import os
import pandas as pd

def process_cla_gt_labels(cla_gt_folder_path, cla_gt_order_file_path, cla_gt_output_csv_path):
    """
    处理 .txt 文件，提取标签并将其保存为 CSV 文件。如果输出文件夹或文件不存在，则自动创建。

    参数:
    cla_gt_folder_path (str): .txt 文件所在的文件夹路径。
    cla_gt_order_file_path (str): order CSV 文件路径。
    cla_gt_output_csv_path (str): 输出的 CSV 文件路径。
    """
    # 检查输出文件夹是否存在，如果不存在则创建
    output_folder = os.path.dirname(cla_gt_output_csv_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # 读取 order CSV 文件
    order_df = pd.read_csv(cla_gt_order_file_path, encoding='utf-8')

    # 假设第一列是序号，第二列是文件名
    # 提取序号和文件名
    order_df['filename'] = order_df.iloc[:, 1].apply(lambda x: x.zfill(4))  # 将文件名补齐为四位
    order_df['index'] = order_df.index + 1  # 添加序号列，从 1 开始

    # 初始化一个空的列表来存储提取的数据
    data = []

    # 遍历 order 列表中的每个文件名
    for idx, row in order_df.iterrows():
        file_name = row['filename']  # 获取四位文件名
        file_path = os.path.join(cla_gt_folder_path, f'{file_name}.txt')  # 根据文件名生成对应的文件路径

        # 检查文件是否存在
        if os.path.exists(file_path):
            # 读取 .txt 文件的第一行
            with open(file_path, 'r', encoding='utf-8') as file:
                first_line = file.readline().strip()  # 读取第一行并去掉前后空格

                # 提取第一个数字并减一
                first_number = int(first_line.split()[0]) - 1  # 提取第一个元素并减一

                # 将数据存入列表，第一列为序号，第二列为减去 1 后的数字
                data.append({'id': row['index'], 'label': first_number})
        else:
            print(f"File {file_name}.txt does not exist.")

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)

    # 保存为 .csv 文件
    df.to_csv(cla_gt_output_csv_path, index=False, encoding='utf-8')

    print(f"Data saved to {cla_gt_output_csv_path}")


