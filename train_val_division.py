import os
import random
import shutil

# 设置随机数种子，以确保每次运行的结果相同
random.seed(42)

# 定义训练集和验证集比例
train_ratio = 0.8
val_ratio = 0.2

# 定义数据集目录和输出目录
dataset_dir = "D:\\地震相数据集\\parihaka_new\\训练集\\地震相图\\X方向剪裁jpg"
output_dir = "C:\\Users\\63037\\Desktop\\jpg"

# 获取所有图片文件的路径
image_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".jpg")]

# 随机打乱图片文件列表
random.shuffle(image_files)

# 计算训练集和验证集的数量
num_train = int(len(image_files) * train_ratio)
num_val = int(len(image_files) * val_ratio)

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 将前num_train个图片复制到训练集目录
train_dir = os.path.join(output_dir, "train")
os.makedirs(train_dir, exist_ok=True)
for i in range(num_train):
    shutil.copy(image_files[i], train_dir)

# 将接下来的num_val个图片复制到验证集目录
val_dir = os.path.join(output_dir, "val")
os.makedirs(val_dir, exist_ok=True)
for i in range(num_train, num_train+num_val):
    shutil.copy(image_files[i], val_dir)