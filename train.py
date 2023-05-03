import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from models import Teacher, Student
from utils import get_loaders, save_checkpoint, load_checkpoint
from metrics import get_metrics
from tqdm import tqdm
from loss import distil_loss



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_IMG_DIR = './data/train/'
VAL_IMG_DIR = './data/test/'
TRAIN_LABEL_PATH = './data/train_labels.csv'
VAL_LABEL_PATH = './data/test_labels.csv'
BATCH_SIZE = 512
EPOCHS = 20
LR = 0.001
NUM_CLASSES = 10

USE_TEACHER = False
FROM_CHECKPOINT = False
CHECKPOINT_PATH = None
TEACHER_CHECKPOINT_PATH = 'checkpoints/teacher/final.pth.tar'
TENSORBOARD_PATH = ('./tensorboard/train/without_teacher', './tensorboard/train/with_teacher')[USE_TEACHER]



def train_step(student, teacher, loader, optimizer, loss_func):
    train_total_loss = 0
    for batch_idx, (data, labels) in enumerate(tqdm(loader)):
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)

        pred = student(data).to(DEVICE)

        if USE_TEACHER:
            teacher_pred = teacher(data).to(DEVICE)
            loss, soft, hard = loss_func(preds=pred, real_labels=labels, teacher_labels=teacher_pred)
        else:
            loss = loss_func(pred, labels)
  
        train_total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_total_loss / (batch_idx+1)



def val_step(student, loader, loss_func):
    val_total_loss = 0
    val_total_metrics = torch.Tensor([0, 0, 0])
    student.eval()

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(loader):
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)

            pred = student(data)

            loss = loss_func(pred, labels)
            val_total_loss += loss

            pred = F.softmax(pred, dim=1)
            val_total_metrics += get_metrics(pred, labels, DEVICE)

    student.train()
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

    student = Student(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(params=student.parameters(), lr=LR)
    if FROM_CHECKPOINT:
        load_checkpoint(path=CHECKPOINT_PATH, model=student, optimizer=optimizer)

    val_loss = nn.CrossEntropyLoss()

    if USE_TEACHER:
        teacher = Teacher(num_classes=NUM_CLASSES, path=TEACHER_CHECKPOINT_PATH).to(DEVICE)
        teacher.eval()
        loss_func = distil_loss
    else:
        teacher = None
        loss_func = val_loss

    writer = SummaryWriter(TENSORBOARD_PATH)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, verbose=True)

    for epoch in range(EPOCHS):
        train_loss = train_step(student, teacher, train_loader, optimizer, loss_func)
        print(f'==> Epoch {epoch}: train loss = {train_loss}')
        # scheduler.step(train_loss)

        val_stat = val_step(student, val_loader, val_loss)
        print(f'\tValidation: Loss = {round(val_stat["val_loss"].item(), 5)}, Accuracy = {round(val_stat["acc"].item(),5)}, \
F1 = {round(val_stat["f1"].item(), 5)}, ROC-AUC = {round(val_stat["roc"].item(), 5)}')
              
        writer.add_scalar('Training loss', train_loss, global_step=epoch)
        writer.add_scalar('Validation loss', val_stat['val_loss'].item(), global_step=epoch)
        writer.add_scalar('Accuracy', val_stat['acc'].item(), global_step=epoch)
        writer.add_scalar('F1', val_stat['f1'].item(), global_step=epoch)
        writer.add_scalar('ROC-AUC', val_stat['roc'].item(), global_step=epoch)

#         if epoch % 50 == 0:
#             save_checkpoint(student, f'./checkpoints/Epoch{epoch}.pth.tar', optimizer)

#         save_checkpoint(student, f'./checkpoints/final.pth.tar', optimizer)
    

if __name__ == '__main__':
    train()