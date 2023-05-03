import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import MyDataset
from torch.utils.data import DataLoader
import pandas as pd
import torch


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])


def get_val_transform():
    return A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        A.pytorch.ToTensorV2()
    ])


def get_loaders(train_img_dir, val_img_dir, train_label_path, val_label_path, batch_size, num_workers=2, pin_memory=True):
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_set = MyDataset(train_img_dir, train_label_path, transform=train_transform)
    val_set = MyDataset(val_img_dir, val_label_path, transform=val_transform)

    train_loader = DataLoader(
        dataset=train_set, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        dataset=val_set, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        shuffle=False
    )

    return train_loader, val_loader



def enocde_labels(train_label_path, val_label_path):
    train = pd.read_csv(train_label_path)
    val = pd.read_csv(val_label_path)

    unique = set(train['label'].values).union(set(val['label'].values))
    encoder = dict(zip(unique, range(len(unique))))

    train['label'] = train['label'].apply(lambda x: encoder[x])
    val['label'] = val['label'].apply(lambda x: encoder[x])

    train.to_csv(train_label_path, index=False)
    val.to_csv(val_label_path, index=False)

    return encoder



def save_checkpoint(model, path, optimizer=None):
    print("==> Saving checkpoint")
    checkpoint = {"state_dict": model.state_dict()}
    if optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer):
    print("==> Loading checkpoint")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])