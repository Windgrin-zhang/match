# import os
# import pandas as pd
#
# # 设置要遍历的文件夹路径
# folder_path = './test_A/A_test_cla/Real_A/labels'  # 替换为你的文件夹路径
# output_csv_path = './runs/res_csv/cla_gt.csv'  # 输出 CSV 文件的路径
#
# # 初始化一个空的列表来存储提取的数据
# data = []
#
# # 初始化计数器
# counter = 1
#
# # 遍历文件夹中的所有 .txt 文件
# for filename in os.listdir(folder_path):
#     if filename.endswith('.txt'):
#         file_path = os.path.join(folder_path, filename)
#
#         # 读取 .txt 文件的第一行
#         with open(file_path, 'r', encoding='utf-8') as file:
#             first_line = file.readline().strip()  # 读取第一行并去除前后空格
#
#             # 提取第一个数字
#             first_number = first_line.split()[0]  # 按空格分割并提取第一个元素
#
#             # 将数据存入列表，文件名替换为计数器的值
#             data.append({'filename': f'{counter}', 'first_number': first_number})
#
#             # 增加计数器
#             counter += 1
#
# # 将数据转换为 DataFrame
# df = pd.DataFrame(data)
#
# # 保存为 .csv 文件
# df.to_csv(output_csv_path, index=False, encoding='utf-8')
#
# print(f"Data saved to {output_csv_path}")



import os
import pandas as pd

# 设置文件夹路径和 order 文件路径
folder_path = './test_A/A_test_cla/Real_A/labels'  # 替换为你的 .txt 文件所在的文件夹路径
order_file_path = './runs/csv/cla_order.csv'  # 替换为你的 order 文件路径
output_csv_path = './runs/csv/cla_gt.csv'  # 输出 CSV 文件的路径

# 读取 order CSV 文件
order_df = pd.read_csv(order_file_path, encoding='utf-8')

# 假设第一列是序号，第二列是文件名
# 提取序号和文件名
order_df['filename'] = order_df.iloc[:, 1].apply(lambda x: x.zfill(4))  # 将文件名补齐为四位
order_df['index'] = order_df.index + 1  # 添加序号列，从 1 开始

# 初始化一个空的列表来存储提取的数据
data = []

# 遍历 order 列表中的每个文件名
for idx, row in order_df.iterrows():
    file_name = row['filename']  # 获取四位文件名
    file_path = os.path.join(folder_path, f'{file_name}.txt')  # 根据文件名生成对应的文件路径

    # 检查文件是否存在
    if os.path.exists(file_path):
        # 读取 .txt 文件的第一行
        with open(file_path, 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()  # 读取第一行并去掉前后空格

            # 提取第一个数字
            first_number = first_line.split()[0]  # 按空格分割并提取第一个元素

            # 将数据存入列表，第一列为序号，第二列为第一个数字
            data.append({'index': row['index'], 'first_number': first_number})
    else:
        print(f"File {file_name}.txt does not exist.")

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 保存为 .csv 文件
df.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"Data saved to {output_csv_path}")
