import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class Teacher(nn.Module):
    def __init__(self, num_classes, path=None):
        super().__init__()

        self.model = torchvision.models.resnet152( pretrained=(not bool(path)) )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if path:
            checkpoint = torch.load(path)['state_dict']
            for key in list(checkpoint.keys()):
                checkpoint[key.replace('model.', '')] = checkpoint.pop(key)
            self.model.load_state_dict(checkpoint)
            for param in self.model.parameters():
                param.requires_grad = False


    def forward(self, x):
        x = self.model(x)
        return x



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x = self.block(x)
        return x
    


class Student(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ConvBlock(in_channels, 8),
                ConvBlock(8, 16),
                ConvBlock(16, 32),
            ]
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512, 100)
        self.fc2 = nn.Linear(100, 10)

    
    def forward(self, x):

        for block in self.blocks:
            x = block(x)
            x = self.pool(x)

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
