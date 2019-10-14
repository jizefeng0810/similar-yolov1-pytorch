import torch
import torch.nn as nn
import torch.nn.functional as F

class V1_Loss(nn.Module):
    def __init__(self):
       super(V1_Loss, self).__init__()
    def forward(self, pred_tensor, target_tensor):
        # 具有目标的标签逻辑索引
        pred_loc_data, pred_con_data= pred_tensor.split([4,2],dim=3)
        target_loc_data, target_con_data = target_tensor.split([4, 2], dim=3)

        pos = target_tensor[:, :, :, 5] > 0

        # location loss
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(target_loc_data)
        loc_p = pred_loc_data[pos_idx].view(-1, 4)
        loc_t = target_loc_data[pos_idx].view(-1, 4)
        loss_smooth_l1 = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # classification loss
        y_hat = F.softmax(pred_con_data, dim=-1)
        y = target_con_data
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(target_con_data)
        pos_clf_loss = torch.sum((- y * torch.log(y_hat)) * pos_idx.float())
        neg_clf_loss = torch.sum((- y * torch.log(y_hat)) * (1. - pos_idx.float()))
        loss_cross_entropy = pos_clf_loss + 0.025*neg_clf_loss


        return 10. * loss_smooth_l1 + loss_cross_entropy


