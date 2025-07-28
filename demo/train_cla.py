# # import optuna
# # from ultralytics import YOLO
# # from huizong_cla__gt import process_cla_gt_labels
# # from huizong import copy_images_and_first_line_of_txt_to_directories
# # from huizong import copy_random_images_with_labels
# # from change_labels import modify_first_number_in_each_line
# # from huizongA import copy_images_and_first_line_of_txt_to_directories_A
# #
# #
# # # 将train_cla中的所有图片和标签汇总到Real_train中
# # # 指定源目录和目标目录（相对路径或绝对路径）
# # train_cla_source_dir = './train_cla/train'
# # Real_train_image_target_dir = './train_cla/Real_train/images'
# # Real_train_txt_target_dir = './train_cla/Real_train/labels'
# #
# # copy_images_and_first_line_of_txt_to_directories(train_cla_source_dir, Real_train_image_target_dir, Real_train_txt_target_dir)
# #
# # # 将Real_train中的文件随机选出十分之一作为验证集
# # # 指定源图片和标签目录以及目标图片和标签目录
# # val_source_image_dir = './train_cla/Real_train/images'
# # val_source_label_dir = './train_cla/Real_train/labels'
# # val_target_image_dir = './train_cla/val/images'
# # val_target_label_dir = './train_cla/val/labels'
# #
# # # 调用函数，选择并复制十分之一的图片和标签
# # copy_random_images_with_labels(val_source_image_dir, val_source_label_dir, val_target_image_dir, val_target_label_dir, fraction=0.1)
# #
# # # 使得标签减一
# # # # 指定包含txt文件的目录
# #
# # # 指定包含txt文件的目录
# # directory_path = './train_cla/Real_train/labels'
# # modify_first_number_in_each_line(directory_path)
# #
# # # 指定包含txt文件的目录
# # directory_path = './train_cla/val/labels'
# # modify_first_number_in_each_line(directory_path)
# #
# # # 汇总test_A中的A_test_cla中的所有图片文件
# # # 指定源目录和目标目录（相对路径或绝对路径）
# # source_dir = './test_A/A_test_cla/A'
# # image_target_dir = './test_A/A_test_cla/Real_A/images'
# # txt_target_dir = './test_A/A_test_cla/Real_A/labels'
# #
# # copy_images_and_first_line_of_txt_to_directories_A(source_dir, image_target_dir, txt_target_dir)
# #
# #
# # # 汇总作出gt文件
# # cla_gt_folder_path = './test_A/A_test_cla/Real_A/labels'
# # cla_gt_order_file_path = './runs/csv/cla_order.csv'
# # cla_gt_output_csv_path = './runs/csv/cla_gt.csv'
# #
# # process_cla_gt_labels(cla_gt_folder_path, cla_gt_order_file_path,cla_gt_output_csv_path)
# #
# #
# #
# # # 加载模型函数
# # def load_model():
# #     model = YOLO("yolov8n.yaml")  # 如果要从头开始构建模型
# #     model = YOLO("yolov8n.pt")  # 加载预训练模型
# #     return model
# #
# #
# # # 目标函数，用于贝叶斯优化
# # def objective(trial):
# #     try:
# #         learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
# #         hsv_s = trial.suggest_float('hsv_s', 0.3, 0.9)
# #         contrast = trial.suggest_float('contrast', 0.3, 0.9)
# #
# #         model = load_model()
# #
# #         model.train(
# #             data="A_fea.yaml",
# #             epochs=10,
# #             batch=4,
# #             augment=True,
# #             hsv_s=hsv_s,
# #             contrast=contrast,
# #             lr0=learning_rate
# #         )
# #
# #         metrics = model.val()
# #         print(metrics)  # 打印metrics以便于调试
# #         return metrics.maps[0]  # 假设mAP在maps属性中，取第一个值
# #
# #     except Exception as e:
# #         print(f"Error during trial: {e}")
# #         return 0.0  # 返回默认值以避免Optuna异常
# #
# #
# #
# # # 贝叶斯优化主函数
# # def run_optimization():
# #     # 创建Optuna实验并运行优化
# #     study = optuna.create_study(direction='maximize')
# #     study.optimize(objective, n_trials=20)  # 设置优化次数
# #
# #     # 输出最佳参数
# #     print("最佳参数:", study.best_params)
# #     print("最佳mAP:", study.best_value)
# #
# #     # 将最佳参数保存到文件中
# #     with open("best_params.txt", "w") as f:
# #         f.write(f"最佳参数: {study.best_params}\n")
# #         f.write(f"最佳mAP: {study.best_value}\n")
# #
# #     return study.best_params
# #
# #
# # # 主程序入口
# # if __name__ == '__main__':
# #     # 运行贝叶斯优化，获得最优参数
# #     best_params = run_optimization()
# #
# #     # 使用最优参数进行完整训练
# #     model = load_model()
# #     model.train(
# #         data="A_cla.yaml",
# #         epochs=50,  # 完整训练周期
# #         batch=4,  # 批次大小
# #         augment=True,
# #         hsv_s=best_params['hsv_s'],  # 使用优化得到的饱和度参数
# #         # contrast=best_params['contrast'],  # 使用优化得到的对比度参数
# #         lr0=best_params['learning_rate']  # 使用优化得到的学习率
# #     )
# #
# #     # 验证模型性能
# #     metrics = model.val()
# #     # 导出模型到ONNX格式
# #     path = model.export(format="onnx")
#
# if __name__ == '__main__':
# # Use the model
#     model.train(data="A_cla.yaml", epochs=50,batch = 8)  # train the model
#     metrics = model.val(data="A_cla.yaml")   # evaluate model performance on the validation set
#     path = model.export(format="onnx")  # export the model to ONNX format
#
# import argparse
# import datetime
# from itertools import chain
# import os
# from pathlib import Path
# import shutil
# import yaml
# import pandas as pd
# from collections import Counter
# from sklearn.model_selection import KFold
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor
# from ultralytics import YOLO
#
# #
# # # 导入其他模块和自定义函数
# # from huizong_cla__gt import process_cla_gt_labels
# # from huizong import copy_images_and_first_line_of_txt_to_directories
# # from huizong import copy_random_images_with_labels
# # from change_labels import modify_first_number_in_each_line
# # from huizongA import copy_images_and_first_line_of_txt_to_directories_A
# #
# #
# # # 将train_cla中的所有图片和标签汇总到Real_train中
# # # 指定源目录和目标目录（相对路径或绝对路径）
# # train_cla_source_dir = './train_cla/train'
# # Real_train_image_target_dir = './train_cla/Real_train/images'
# # Real_train_txt_target_dir = './train_cla/Real_train/labels'
# #
# # copy_images_and_first_line_of_txt_to_directories(train_cla_source_dir, Real_train_image_target_dir, Real_train_txt_target_dir)
# #
# # # 将Real_train中的文件随机选出十分之一作为验证集
# # # 指定源图片和标签目录以及目标图片和标签目录
# # val_source_image_dir = './train_cla/Real_train/images'
# # val_source_label_dir = './train_cla/Real_train/labels'
# # val_target_image_dir = './train_cla/val/images'
# # val_target_label_dir = './train_cla/val/labels'
# #
# # # 调用函数，选择并复制十分之一的图片和标签
# # copy_random_images_with_labels(val_source_image_dir, val_source_label_dir, val_target_image_dir, val_target_label_dir, fraction=0.1)
# #
# # # 使得标签减一
# # # # 指定包含txt文件的目录
# #
# # # 指定包含txt文件的目录
# # directory_path = './train_cla/Real_train/labels'
# # modify_first_number_in_each_line(directory_path)
# #
# # # 指定包含txt文件的目录
# # directory_path = './train_cla/val/labels'
# # modify_first_number_in_each_line(directory_path)
# #
# # # 汇总test_A中的A_test_cla中的所有图片文件
# # # 指定源目录和目标目录（相对路径或绝对路径）
# # source_dir = './test_A/A_test_cla/A'
# # image_target_dir = './test_A/A_test_cla/Real_A/images'
# # txt_target_dir = './test_A/A_test_cla/Real_A/labels'
# #
# # copy_images_and_first_line_of_txt_to_directories_A(source_dir, image_target_dir, txt_target_dir)
# #
# #
# # # 汇总作出gt文件
# # cla_gt_folder_path = './test_A/A_test_cla/Real_A/labels'
# # cla_gt_order_file_path = './runs/csv/cla_order.csv'
# # cla_gt_output_csv_path = './runs/csv/cla_gt.csv'
# #
# # process_cla_gt_labels(cla_gt_folder_path, cla_gt_order_file_path,cla_gt_output_csv_path)
# # #
#
#
# NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
#
# def parse_opt():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--data', default=r'./data')  # 数据集路径
#     parser.add_argument('--ksplit', default=10, type=int)  # K-Fold交叉验证拆分数据集
#     parser.add_argument('--im_suffixes', default=['jpg', 'png', 'jpeg'], help='images suffix')  # 图片后缀名
#
#     return parser.parse_args()
#
# def run(func, this_iter, desc="Processing"):
#     with ThreadPoolExecutor(max_workers=NUM_THREADS, thread_name_prefix='MyThread') as executor:
#         results = list(
#             tqdm(executor.map(func, this_iter), total=len(this_iter), desc=desc)
#         )
#     return results
#
#
# def main(opt):
#     dataset_path, ksplit, im_suffixes = Path(opt.data), opt.ksplit, opt.im_suffixes
#
#     save_path = Path(dataset_path / f'{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-Valid')
#     save_path.mkdir(parents=True, exist_ok=True)
#
#     # 获取所有图像和标签文件的列表
#     images = sorted(list(chain(*[(dataset_path / "images").rglob(f'*.{ext}') for ext in im_suffixes])))
#     labels = sorted((dataset_path / "labels").rglob("*.txt"))
#
#     root_directory = Path.cwd()
#     print("当前文件运行根目录:", root_directory)
#     if len(images) != len(labels):
#         print('*' * 20)
#         print('当前数据集和标签数量不一致！！！')
#         print('*' * 20)
#
#     # 从YAML文件加载类名
#     classes_file = dataset_path / 'classes.yaml'
#     if not classes_file.exists():
#         # 自动生成 classes.yaml 文件
#         class_names = set()
#         for label in labels:
#             with open(label, 'r') as lf:
#                 lines = lf.readlines()
#                 for l in lines:
#                     class_id = int(l.split(' ')[0])
#                     class_names.add(class_id)
#
#         # 将 class_ids 转换为列表
#         class_names = list(class_names)
#
#         # 写入 classes.yaml 文件
#         with open(classes_file, 'w', encoding="utf8") as f:
#             yaml.safe_dump({'names': class_names}, f)
#         print(f"已生成 classes.yaml 文件，包含类别：{class_names}")
#
#     # 读取类名
#     with open(classes_file, 'r', encoding="utf8") as f:
#         classes = yaml.safe_load(f)['names']
#
#     # cls_idx 直接使用类名列表
#     cls_idx = sorted(classes)
#
#     # 创建DataFrame来存储每张图像的标签计数
#     indx = [l.stem for l in labels]  # 使用基本文件名作为ID（无扩展名）
#     labels_df = pd.DataFrame([], columns=cls_idx, index=indx)
#
#     # 计算每张图像的标签计数
#     for label in labels:
#         lbl_counter = Counter()
#         with open(label, 'r') as lf:
#             lines = lf.readlines()
#         for l in lines:
#             lbl_counter[int(l.split(' ')[0])] += 1
#         labels_df.loc[label.stem] = lbl_counter
#
#     # 用0.0替换NaN值
#     labels_df = labels_df.fillna(0.0)
#
#     # K-Fold 交叉验证
#     kf = KFold(n_splits=ksplit, shuffle=True, random_state=10)
#     kfolds = list(kf.split(labels_df))
#     folds = [f'split_{n}' for n in range(1, ksplit + 1)]
#     folds_df = pd.DataFrame(index=indx, columns=folds)
#
#     # 为每个折叠分配图像到训练集或验证集
#     for idx, (train, val) in enumerate(kfolds, start=1):
#         folds_df[f'split_{idx}'].loc[labels_df.iloc[train].index] = 'train'
#         folds_df[f'split_{idx}'].loc[labels_df.iloc[val].index] = 'val'
#
#     # # 写入 yaml_paths.txt
#     # with open(Path(opt.data) / 'yaml_paths.txt', 'w') as f:
#     #     for idx in range(opt.ksplit):
#     #         f.write(
#     #             f'data/{datetime.date.today().isoformat()}_{opt.ksplit}-Fold_Cross-Valid/split_{idx + 1}/dataset.yaml\n')
#
#     # 计算每个折叠的标签分布比例
#     fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)
#     for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
#         train_totals = labels_df.iloc[train_indices].sum()
#         val_totals = labels_df.iloc[val_indices].sum()
#
#         # 为避免分母为零，向分母添加一个小值（1E-7）
#         ratio = val_totals / (train_totals + 1E-7)
#         fold_lbl_distrb.loc[f'split_{n}'] = ratio
#
#     ds_yamls = []
#
#     for split in folds_df.columns:
#         split_dir = save_path / split
#         split_dir.mkdir(parents=True, exist_ok=True)
#         (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
#         (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
#         (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
#         (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
#
#         dataset_yaml = split_dir / f'{split}_dataset.yaml'
#         ds_yamls.append(dataset_yaml.as_posix())
#         split_dir = (root_directory / split_dir).as_posix()
#
#         with open(dataset_yaml, 'w') as ds_y:
#             yaml.safe_dump({
#                 'train': split_dir + '/train/images',
#                 'val': split_dir + '/val/images',
#                 'names': classes
#             }, ds_y)
#     # print(ds_yamls)
#     with open(dataset_path / 'yaml_paths.txt', 'w') as f:
#         for path in ds_yamls:
#             f.write(path + '\n')
#
#     args_list = [(image, save_path, folds_df) for image in images]
#
#     run(split_images_labels, args_list, desc=f"Creating dataset")
#
#
#
# # def main(opt):
# #     dataset_path, ksplit, im_suffixes = Path(opt.data), opt.ksplit, opt.im_suffixes
# #
# #     save_path = Path(dataset_path / f'{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-Valid')
# #     save_path.mkdir(parents=True, exist_ok=True)
# #
# #     # 获取所有图像和标签文件的列表
# #     images = sorted(list(chain(*[(dataset_path / "images").rglob(f'*.{ext}') for ext in im_suffixes])))
# #     # images = sorted(image_files)
# #     labels = sorted((dataset_path / "labels").rglob("*.txt"))
# #
# #     root_directory = Path.cwd()
# #     print("当前文件运行根目录:", root_directory)
# #     if len(images) != len(labels):
# #         print('*' * 20)
# #         print('当前数据集和标签数量不一致！！！')
# #         print('*' * 20)
# #
# #     # 从YAML文件加载类名
# #     classes_file = sorted(dataset_path.rglob('classes.yaml'))[0]
# #     assert classes_file.exists(), "请创建classes.yaml类别文件"
# #     if classes_file.suffix == ".txt":
# #         pass
# #     elif classes_file.suffix == ".yaml":
# #         with open(classes_file, 'r', encoding="utf8") as f:
# #             classes = yaml.safe_load(f)['names']
# #     cls_idx = sorted(classes.keys())
# #
# #     # 创建DataFrame来存储每张图像的标签计数
# #     indx = [l.stem for l in labels]  # 使用基本文件名作为ID（无扩展名）
# #     labels_df = pd.DataFrame([], columns=cls_idx, index=indx)
# #
# #     # 计算每张图像的标签计数
# #     for label in labels:
# #         lbl_counter = Counter()
# #         with open(label, 'r') as lf:
# #             lines = lf.readlines()
# #         for l in lines:
# #             # YOLO标签使用每行的第一个位置的整数作为类别
# #             lbl_counter[int(l.split(' ')[0])] += 1
# #         labels_df.loc[label.stem] = lbl_counter
# #
# #     # 用0.0替换NaN值
# #     labels_df = labels_df.fillna(0.0)
# #
# #     kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # 设置random_state以获得可重复的结果
# #     kfolds = list(kf.split(labels_df))
# #     folds = [f'split_{n}' for n in range(1, ksplit + 1)]
# #     folds_df = pd.DataFrame(index=indx, columns=folds)
# #
# #     # 为每个折叠分配图像到训练集或验证集
# #     for idx, (train, val) in enumerate(kfolds, start=1):
# #         folds_df[f'split_{idx}'].loc[labels_df.iloc[train].index] = 'train'
# #         folds_df[f'split_{idx}'].loc[labels_df.iloc[val].index] = 'val'
# #
# #     # 计算每个折叠的标签分布比例
# #     fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)
# #     for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
# #         train_totals = labels_df.iloc[train_indices].sum()
# #         val_totals = labels_df.iloc[val_indices].sum()
# #
# #         # 为避免分母为零，向分母添加一个小值（1E-7）
# #         ratio = val_totals / (train_totals + 1E-7)
# #         fold_lbl_distrb.loc[f'split_{n}'] = ratio
# #
# #     ds_yamls = []
# #
# #     for split in folds_df.columns:
# #         split_dir = save_path / split
# #         split_dir.mkdir(parents=True, exist_ok=True)
# #         (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
# #         (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
# #         (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
# #         (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
# #
# #         dataset_yaml = split_dir / f'{split}_dataset.yaml'
# #         ds_yamls.append(dataset_yaml.as_posix())
# #         split_dir = (root_directory / split_dir).as_posix()
# #
# #         with open(dataset_yaml, 'w') as ds_y:
# #             yaml.safe_dump({
# #                 'train': split_dir + '/train/images',
# #                 'val': split_dir + '/val/images',
# #                 'names': classes
# #             }, ds_y)
# #     # print(ds_yamls)
# #     with open(dataset_path / 'yaml_paths.txt', 'w') as f:
# #         for path in ds_yamls:
# #             f.write(path + '\n')
# #
# #     args_list = [(image, save_path, folds_df) for image in images]
# #
# #     run(split_images_labels, args_list, desc=f"Creating dataset")
#
#
# def split_images_labels(args):
#     image, save_path, folds_df = args
#     label = image.parents[1] / 'labels' / f'{image.stem}.txt'
#     if label.exists():
#         for split, k_split in folds_df.loc[image.stem].items():
#             # 目标目录
#             img_to_path = save_path / split / k_split / 'images'
#             lbl_to_path = save_path / split / k_split / 'labels'
#             shutil.copy(image, img_to_path / image.name)
#             shutil.copy(label, lbl_to_path / label.name)
#
#
# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)
#
#     # 载入模型和K-Fold训练
#     model = YOLO('yolov8n.pt', task='train')
#
#     # 从yaml_paths.txt中加载K-Fold配置的路径
#     ds_yamls = []
#     with open(Path(opt.data) / 'yaml_paths.txt', 'r') as f:
#         for line in f:
#             line = line.strip()
#             ds_yamls.append(line)
#
#     print(ds_yamls)
#
#     for k in range(opt.ksplit):
#         dataset_yaml = ds_yamls[k]
#         name = Path(dataset_yaml).stem
#         model.train(
#             data=dataset_yaml,
#             batch=16,
#             epochs=50,
#             imgsz=640,
#             device=0,
#             workers=8,
#             project="runs/train",
#             name=name,
#             degrees=10,  # 加入旋转增强
#             translate=0.1,  # 加入平移增强
#             # 你可以添加更多的增强参数
#         )
#
#     # 使用最后一个Fold的数据集进行验证和预测
#     metrics = model.val(data=ds_yamls[-1])  # 使用最后一个fold的验证集
#     print("验证结果:", metrics)
#
#     # 导出模型为ONNX格式
#     path = model.export(format="onnx")
#     print(f"模型已导出到 {path}")
#
#     print("*" * 40)
#     print("K-Fold Cross Validation Completed.")
#     print("*" * 40)
from ultralytics import YOLO


# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)



if __name__ == '__main__':
# Use the model

    model.train(data="A_cla.yaml", epochs = 50,batch = 8,hsv_s = 0.686758843118897)
    metrics = model.val()  # evaluate model performance on the validation set
    path = model.export(format="onnx")  # export the model to ONNX format