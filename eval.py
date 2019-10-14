import Config
import torch
import numpy as np
from data.dataset_yolo import Dataset_yolo
import torchvision.transforms as transforms
from net.resnet_yolo import resnet50
import torch.nn.functional as F


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model_path = './net/trainpath_inria.pth'
val_txt_path = './data/inria_val_data.txt'
val_image_path = './data/inria_person/PICTURES_LABELS_TEST/PICTURES/'
batch_size = Config.batch_size
num_class = Config.num_class
classes_dict = Config.classes_dict

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n, 4)
    :param set_2: set 2, a tensor of dimensions (n, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    # (x1,y1) (x2,y2)
    lower_bounds = torch.max(set_1[:2]-set_1[2:]/2.0, set_2[:2]-set_2[2:]/2.0)
    upper_bounds = torch.min(set_1[:2]+set_1[2:]/2.0, set_2[:2]+set_2[2:]/2.0)  # (2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (2)
    return intersection_dims[0] * intersection_dims[1]  # (n)

def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets   W*H
    areas_set_1 = set_1[2] * set_1[3]  # (n1)
    areas_set_2 = set_2[2] * set_2[3]  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1 + areas_set_2 - intersection  # (n,1)

    return intersection / union  # (n)

def calculate_mAP(det_boxes, det_labels, true_boxes, true_labels):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(true_boxes) == len(true_labels)  # these are all lists of tensors of the same length, i.e. number of images

    det_boxes = torch.reshape(det_boxes,[-1,4])
    det_labels = torch.reshape(det_labels,[-1,2])
    true_boxes = torch.reshape(true_boxes, [-1, 4])
    true_labels = torch.reshape(true_labels, [-1, 2])

    tp = np.zeros(len(true_labels))
    fp = np.zeros(len(true_labels))
    for i in range(len(true_labels)):
        if det_labels[i][1] > 0.85:
            iou = find_jaccard_overlap(det_boxes[i], true_boxes[i])
            if iou > 0.5:
                tp[i] = 1
            else:
                fp[i] = 1
    tp = np.sum(tp)
    fp = np.sum(fp)

    average_precisions = tp / (tp + fp)

    return average_precisions


if __name__ == '__main__':
    val_dataset = Dataset_yolo(image_root=val_image_path, list_file=val_txt_path, train=False, transform=[transforms.ToTensor()])
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,  # 读取数据集
                                             batch_size=batch_size,
                                             shuffle=False)

    model = resnet50().to(device)  # 搭建网络
    trained_net = torch.load(model_path)
    model.load_state_dict(trained_net)
    model.eval()

    with torch.no_grad():
        for step, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            pred = model(images)

            pred_boxes = pred[:,:,:,:4]
            pred_labels = pred[:,:,:,4:]
            pred_labels = F.softmax(pred_labels, dim=-1)
            target_boxes = target[:,:,:,:4]
            target_labels = target[:,:,:,4:]

            if step==0:
                det_boxs = pred_boxes
                det_labels = pred_labels
                true_boxes = target_boxes
                true_labels = target_labels
            else:
                det_boxs = torch.cat((det_boxs,pred_boxes),dim=0)
                det_labels = torch.cat((det_labels, pred_labels), dim=0)
                true_boxes = torch.cat((true_boxes, target_boxes), dim=0)
                true_labels = torch.cat((true_labels, target_labels), dim=0)
        AP = calculate_mAP(det_boxs, det_labels, true_boxes, true_labels)

        print('\nAverage Precision (AP): %.3f' % AP)