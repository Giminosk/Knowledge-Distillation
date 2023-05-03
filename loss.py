import torch
import torch.nn as nn
import torch.nn.functional as F


def distil_loss(preds, real_labels, teacher_labels, writer=None, epoch=None, alpha=0.9, T=5):

    hard_loss = F.cross_entropy(preds, real_labels, reduction='mean')
    
    preds = F.log_softmax(preds / T, dim=1)
    teacher_labels = F.log_softmax(teacher_labels / T, dim=1)
    soft_loss = F.mse_loss(preds, teacher_labels)  * (T ** 2)
    # soft_loss = nn.KLDivLoss(reduction='batchmean')(preds, teacher_labels)

    loss = alpha * soft_loss + (1 - alpha) * hard_loss

    return loss, hard_loss, soft_loss
