import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from data.transforms import Resize, RandomHorizontalFlip, RandomChannelSwap, ToTensor, CenterCrop, RandomRotation, RandomVerticalFlip

resolution_dict = {
    'full' : (384, 1280),
    'tu_small' : (128, 416),
    'tu_big' : (228, 912),
    'half' : (192, 640)}

class KITTIDataset(Dataset):
    def __init__(self, root, split, resolution='full', augmentation='alhashim'):
        self.root = root
        self.split = split
        self.resolution = resolution_dict[resolution]
        self.augmentation = augmentation

        if split=='train':
            self.transform = self.train_transform
            self.root = os.path.join(self.root, 'train')
        elif split=='val':
            self.transform = self.val_transform
            self.root = os.path.join(self.root, 'val')
        elif split=='test':
            if self.augmentation == 'alhashim':
                self.transform = None
            else:
                self.transform = CenterCrop(self.resolution)

            self.root = os.path.join(self.root, 'test')

        self.files = os.listdir(self.root)


    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.files[index])

        data = np.load(image_path)
        depth, image = data['depth'], data['image']

        if self.transform is not None:
            data = self.transform(data)

        image, depth = data['image'], data['depth']
        if self.split == 'test':
            image = np.array(image)
            depth = np.array(depth)
        return image, depth

    def __len__(self):
        return len(self.files)


    def train_transform(self, data):
        if self.augmentation == 'alhashim':
            transform = Compose([
                Resize(self.resolution),
                RandomHorizontalFlip(),
                RandomChannelSwap(0.25),
                ToTensor(test=False, maxDepth=80.0)
            ])
        else:
            transform = Compose([
                RandomRotation(4.5),
                CenterCrop(self.resolution),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomChannelSwap(0.25),
                ToTensor(test=False, maxDepth=80.0)
            ])

        data = transform(data)
        return data

    def val_transform(self, data):
        if self.augmentation == 'alhashim':
            transform = Compose([
                Resize(self.resolution),
                ToTensor(test=True, maxDepth=80.0)
            ])
        else:
            transform = Compose([
                CenterCrop(self.resolution),
                ToTensor(test=True, maxDepth=80.0)
            ])

        data = transform(data)
        return data
