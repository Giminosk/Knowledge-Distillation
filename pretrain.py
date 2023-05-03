import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from models import Teacher
from utils import get_loaders, save_checkpoint
from metrics import get_metrics
from tqdm import tqdm


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_IMG_DIR = './data/train/'
VAL_IMG_DIR = './data/test/'
TRAIN_LABEL_PATH = './data/train_labels.csv'
VAL_LABEL_PATH = './data/test_labels.csv'
BATCH_SIZE = 512
EPOCHS = 20
LR = 0.0001
NUM_CLASSES = 10


def train_step(model, loader, optimizer, loss_func):
    train_total_loss = 0
    for batch_idx, (data, labels) in enumerate(tqdm(loader)):
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)

        pred = model(data)

        loss = loss_func(pred, labels)
        train_total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_total_loss / (batch_idx+1)



def val_step(model, loader, loss_func):
    val_total_loss = 0
    val_total_metrics = torch.Tensor([0, 0, 0])
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(loader):
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)

            pred = model(data)

            loss = loss_func(pred, labels)
            val_total_loss += loss

            pred = F.softmax(pred, dim=1)
            val_total_metrics += get_metrics(pred, labels, DEVICE)

    model.train()
    val_total_loss = val_total_loss / (batch_idx+1)
    val_total_metrics = val_total_metrics / (batch_idx+1)

    return {
                'val_loss': val_total_loss,
                'acc': val_total_metrics[0],
                'f1': val_total_metrics[1],
                'roc': val_total_metrics[2]
           } 



def train():

    train_loader, val_loader = get_loaders(
        train_img_dir=TRAIN_IMG_DIR,
        val_img_dir=VAL_IMG_DIR,
        train_label_path=TRAIN_LABEL_PATH,
        val_label_path=VAL_LABEL_PATH,
        batch_size=BATCH_SIZE
    )

    model = Teacher(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(params=model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    writer = SummaryWriter('./tensorboard/pretrain')
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, verbose=True)

    for epoch in range(EPOCHS):
        train_loss = train_step(model, train_loader, optimizer, loss_func)
        print(f'==> Epoch {epoch}: train loss = {train_loss}')
        # scheduler.step(train_loss)

        val_stat = val_step(model, val_loader, loss_func)
        print(f'\tValidation: Loss = {round(val_stat["val_loss"].item(), 5)}, Accuracy = {round(val_stat["acc"].item(),5)}, \
F1 = {round(val_stat["f1"].item(), 5)}, ROC-AUC = {round(val_stat["roc"].item(), 5)}')
              
        writer.add_scalar('Pretrain Training loss', train_loss, global_step=epoch)
        writer.add_scalar('Pretrain Validation loss', val_stat['val_loss'].item(), global_step=epoch)
        writer.add_scalar('Pretrain Accuracy', val_stat['acc'].item(), global_step=epoch)
        writer.add_scalar('Pretrain F1', val_stat['f1'].item(), global_step=epoch)
        writer.add_scalar('Pretrain ROC-AUC', val_stat['roc'].item(), global_step=epoch)
    
    save_checkpoint(model, './checkpoints/teacher/final.pth.tar')


if __name__ == '__main__':
    train()