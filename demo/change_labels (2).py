import os
import re

def modify_first_number_in_each_line(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()  # 读取所有行

            modified_lines = []
            for line in lines:
                # 使用正则表达式替换每一行的第一个数字
                modified_line = re.sub(r'^\s*([1-6])', lambda x: str(int(x.group(0)) - 1), line)
                modified_lines.append(modified_line)

            # 将修改后的内容写回文件
            with open(filepath, 'w', encoding='utf-8') as file:
                file.writelines(modified_lines)

# # 指定包含txt文件的目录

# 指定包含txt文件的目录
directory_path = './train_cla/Real_train/labels'
modify_first_number_in_each_line(directory_path)

# 指定包含txt文件的目录
directory_path = './train_cla/val/labels'
modify_first_number_in_each_line(directory_path)
