"""
input: n * image size(448,448,3)
output: n * 14 * 14 * 6(x,y,w,h,background,person)
"""

import torch
import Config
from torch.utils.data import TensorDataset
from data.dataset_yolo import Dataset_yolo
from net.net_loss import V1_Loss
import os
from math import ceil
import numpy as np
from torchvision import models
from other.logging import logger
from net.resnet_yolo import resnet50
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

epochs = Config.num_epochs
learning_rate = Config.learning_rate
momentum = Config.momentum
weight_decay = Config.weight_decay
num_class = Config.num_class
batch_size = Config.batch_size
cell_size = Config.cell_size
box_num = Config.box_num

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    train_dataset = Dataset_yolo(image_root=Config.train_image_path, list_file=Config.train_txt_path,train=True,transform=[transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,  # 读取数据集
                                               batch_size=batch_size,
                                               shuffle=True, )
    train_num = train_loader.dataset.num_samples
    val_dataset = Dataset_yolo(image_root=Config.val_image_path, list_file=Config.val_txt_path,train=False,transform=[transforms.ToTensor()])
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,       # 读取数据集
                                             batch_size=batch_size,
                                             shuffle=False)
    val_num = val_loader.dataset.num_samples

    model = resnet50().to(device)  # 搭建yolo-v1网络

    """load my trainnet parameters"""
    pretrained_net = torch.load(Config.current_epoch_model_path)
    model.load_state_dict(pretrained_net)

    """set resnet50's pretrained parameters"""
    # resnet = models.resnet50(pretrained=True)
    # new_state_dict = resnet.state_dict()
    # dd = model.state_dict()
    # for k in new_state_dict.keys():
    #     print(k)
    #     if k in dd.keys() and not k.startswith('fc'):
    #         print('yes')
    #         dd[k] = new_state_dict[k]
    # model.load_state_dict(dd)

    criterion = V1_Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=Config.momentum,
                                weight_decay=Config.weight_decay)

    """cal total parameters"""
    total = sum(p.numel() for p in model.parameters())
    print('Parameter Number: ' + str(total))

    # training
    for epoch in range(epochs):
        total_loss = 0.

        # update learning_rate
        if epoch == 300:
            learning_rate = 0.0001
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        if epoch == 500:
            learning_rate = 0.00001
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            output = torch.reshape(output, (len(labels), cell_size, cell_size, (5 * box_num + num_class)))

            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), 1, norm_type=2)

            optimizer.step()

            total_loss = total_loss + float(loss.data)
            if (step + 1) % Config.print_freq == 0:
                logger.info('Epoch-%d/%d:Step-%d/%d:Loss-%.4f:Current Epoch Meanloss-%.4f' % (
                epoch + 1, epochs, step + 1, ceil(train_num / batch_size), loss.data, (total_loss / (step + 1))))
            if epoch % 50 == 0 and epoch > 0 and step == 0:
                torch.save(model.state_dict(), Config.current_epoch_model_path)
                pass
# 验证
    model.eval()
    validation_loss = 0.
    best_test_loss = np.inf
    with torch.no_grad():
        for step, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            pred = model(images)

            loss = criterion(pred, target)
            validation_loss = validation_loss + float(loss.data)
        validation_loss /= val_num
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            logger.info('Best mean Loss-%.5f' % best_test_loss)
            torch.save(model.state_dict(), Config.best_test_loss_model_path)
