import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# 加载CSV文件
cla_gt = pd.read_csv('./runs/csv/cla_gt.csv')
cla_pre = pd.read_csv('./runs/csv/cla_pre.csv')
fea_gt = pd.read_csv('./runs/csv/fea_gt.csv')
fea_pre = pd.read_csv('./runs/csv/fea_pre.csv')

cla_gt.columns = ['id', 'true_label']
cla_pre.columns = ['id', 'pred_label']
fea_gt.columns = ['id', 'true_boundary', 'true_calcification', 'true_direction', 'true_shape']
fea_pre.columns = ['id', 'pred_boundary', 'pred_calcification', 'pred_direction', 'pred_shape']

# 合并结果
cla_merged = pd.merge(cla_gt, cla_pre, on='id')

# 计算分类模型的准确率和平均F1分数
cla_accuracy = accuracy_score(cla_merged['true_label'], cla_merged['pred_label'])
cla_f1 = f1_score(cla_merged['true_label'], cla_merged['pred_label'], average='macro')

# 合并结果
fea_merged = pd.merge(fea_gt, fea_pre, on='id')

# 分别计算每个特征的准确率和F1分数
fea_accuracy_boundary = accuracy_score(fea_merged['true_boundary'], fea_merged['pred_boundary'])
fea_accuracy_calcification = accuracy_score(fea_merged['true_calcification'], fea_merged['pred_calcification'])
fea_accuracy_direction = accuracy_score(fea_merged['true_direction'], fea_merged['pred_direction'])
fea_accuracy_shape = accuracy_score(fea_merged['true_shape'], fea_merged['pred_shape'])

fea_f1_direction = f1_score(fea_merged['true_boundary'], fea_merged['pred_boundary'])
fea_f1_calcification = f1_score(fea_merged['true_calcification'], fea_merged['pred_calcification'])
fea_f1_directiob = f1_score(fea_merged['true_direction'], fea_merged['pred_direction'])
fea_f1_shape = f1_score(fea_merged['true_shape'], fea_merged['pred_shape'])

# 计算平均准确率和平均F1分数
fea_avg_accuracy = (fea_accuracy_boundary + fea_accuracy_calcification + fea_accuracy_direction + fea_accuracy_shape) / 4
fea_avg_f1 = (fea_f1_direction + fea_f1_calcification + fea_f1_directiob + fea_f1_shape) / 4

# 计算最终得分
final_score = 0.3 * (cla_accuracy + fea_avg_accuracy) + 0.2 * (cla_f1 + fea_avg_f1)
final_score = round(final_score * 100, 2)

# 打印结果
print(f"分类模型的准确率: {cla_accuracy}")
print(f"分类模型的平均F1分数: {cla_f1}")
print(f"特征模型的平均准确率: {fea_avg_accuracy}")
print(f"特征模型的平均F1分数: {fea_avg_f1}")
print(f"最终得分: {final_score}")
