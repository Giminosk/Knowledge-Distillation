import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, img_dir, label_path, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.label_path = label_path
        self.transform = transform
        self.labels_df = pd.read_csv(label_path)


    def __getitem__(self, index):
        sample = self.labels_df.iloc[index]
        img_path = os.path.join(self.img_dir, str(sample['id'])+'.png')
        img = np.array(Image.open(img_path).convert('RGB'))
        label = torch.tensor(sample['label'])

        if self.transform:
            img = self.transform(image=img)['image']

        return img, label
    
    def __len__(self):
        return len(self.labels_df)