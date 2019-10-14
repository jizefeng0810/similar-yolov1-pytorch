import cv2
import torch
from net.resnet_yolo import resnet50
import os
import numpy as np
import Config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

path = Config.train_image_path
# path = Config.val_image_path
pretrained_net_path = './other/trainpath_inria.pth'

Color = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]


# start predict one image
#
def predict_gpu(model ,image_path):
    result = []
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(Config.image_size,Config.image_size))
    image = image * 2. / 255. - 1.
    image = torch.Tensor(image).permute(2, 0, 1)
    image = torch.reshape(image, shape=(1, 3, 448, 448))

    print('predicting ' + image_path)
    pred = model(image)  # 1x14x14x(4+2)

    output = torch.reshape(pred, shape=(-1, 6))
    logits = output[:, 4:]
    clf_score = torch.nn.functional.softmax(logits, dim=-1)
    bboxes = output[:, :4].detach().numpy()
    scores = clf_score[:, 1].detach().numpy()

    for box, score in zip(bboxes, scores):
        if score > 0.85:
            x1 = np.int32((box[0] - box[2] / 2.) * 448)
            y1 = np.int32((box[1] - box[3] / 2.) * 448)
            x2 = np.int32((box[0] + box[2] / 2.) * 448)
            y2 = np.int32((box[1] + box[3] / 2.) * 448)
            result.append([(x1, y1), (x2, y2), score])
    return result

if __name__ == '__main__':
    net = resnet50()
    print('load model...')
    pretrained_net = torch.load(pretrained_net_path)
    net.load_state_dict(pretrained_net)
    net.eval()

    image_indexs = os.listdir(path)
    for image_index in image_indexs:
        image_path = path + image_index
        image_show = cv2.imread(image_path)
        image_show = cv2.resize(image_show, (Config.image_size, Config.image_size))

        result = predict_gpu(net, image_path)

        for left_up, right_bottom, score in result:
            cv2.rectangle(image_show ,left_up ,right_bottom ,(64, 128, 0), 2)
            label = Config.classes_name[0] #+str(round(score ,2))
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1 = (left_up[0], left_up[1 ]- text_size[1])
            cv2.rectangle(image_show, (p1[0] - 2// 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                          (64, 128, 0), -1)
            cv2.putText(image_show, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

        cv2.imshow(image_index, image_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('result.jpg', image_show)
