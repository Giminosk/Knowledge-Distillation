import torch
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import MulticlassAUROC


def get_metrics(pred, label, device, num_classes=10):
    acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1 = F1Score(task="multiclass", num_classes=num_classes).to(device)
    roc_auc = MulticlassAUROC(num_classes=num_classes, average="macro", thresholds=None).to(device)
    
    roc_auc_score = roc_auc(pred, label)
    pred = torch.argmax(pred, dim=1)
    acc_score = acc(pred, label)
    f1_score = f1(pred, label)
    
    return torch.Tensor([acc_score, f1_score, roc_auc_score])