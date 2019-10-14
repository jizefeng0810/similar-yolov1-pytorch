# 训练集数据
train_data_path = './data/hazy_person/PICTURES_LABELS_TRAIN/ANOTATION/'
train_image_path = './data/hazy_person/PICTURES_LABELS_TRAIN/PICTURES/'
# 验证集数据
val_data_path = './data/hazy_person/PICTURES_LABELS_TEST/ANOTATION/'
val_image_path = './data/hazy_person/PICTURES_LABELS_TEST/PICTURES/'
# 读取训练集和验证集数据后，打包，避免每次耗时重复读取
# train_path = './data/train.pkl'
# image_train_path = './data/train_images.pkl'
# val_path = './data/val.pkl'
# image_val_path = './data/val_images.pkl'

# hazy_person
train_txt_path = './data/hazy_train_data.txt'
val_txt_path = './data/hazy_val_data.txt'
# inria_person
# train_txt_path = './data/inria_train_data.txt'
# val_txt_path = './data/inria_val_data.txt'

# 图片缩放大小
image_size = 448
# 图片分成7*7块
cell_size = 14
# 是否水平翻转，数据增强
flipped = False
# 迭代次数
num_epochs = 2000
# batch
batch_size = 20
# box number
box_num = 1
# 学习率
learning_rate = 0.001

momentum = 0.9
weight_decay = 5e-4
# 打印信息频率
print_freq = 1


# 目标种类
# classes_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#                 "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes_name = ['person']
# 为目标种类编号
classes_num = [i for i in range(len(classes_name))]
# 种类名对应编号
classes_dict = dict(zip(classes_name, classes_num))
#种类数量
num_class=len(classes_name)

# 存储训练模型数据路径
current_epoch_model_path = './net/trainpath_hazy.pth'
# 存储验证模型数据路径
best_test_loss_model_path = './net/valpath_hazy.pth'
