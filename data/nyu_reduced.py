import pandas as pd
import numpy as np
import torch
import os
from zipfile import ZipFile
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from data.transforms import Resize, RandomHorizontalFlip, RandomChannelSwap, ToTensor

resolution_dict = {
    'full' : (480, 640),
    'half' : (240, 320),
    'mini' : (224, 224)}

class depthDatasetMemory(Dataset):
    def __init__(self, data, split, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform
        self.split = split

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open( BytesIO(self.data[sample[0]]) )
        depth = Image.open( BytesIO(self.data[sample[1]]) )
        image = np.array(image).astype(np.float32)
        depth = np.array(depth).astype(np.float32)

        if self.split == 'train':
            depth = depth /255.0 * 10.0 #From 8bit to range [0, 10] (meter)
        elif self.split == 'val':
            depth = depth * 0.001

        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.nyu_dataset)

class NYU_Testset_Extracted(Dataset):
    def __init__(self, root, resolution='full'):
        self.root = root
        self.resolution = resolution_dict[resolution]

        self.files = os.listdir(self.root)


    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.files[index])

        data = np.load(image_path)
        depth, image = data['depth'], data['image']
        depth = np.expand_dims(depth, axis=2)

        image, depth = data['image'], data['depth']
        image = np.array(image)
        depth = np.array(depth)
        return image, depth

    def __len__(self):
        return len(self.files)



class NYU_Testset(Dataset):
    def __init__(self, zip_path):
        input_zip=ZipFile(zip_path)
        data = {name: input_zip.read(name) for name in input_zip.namelist()}
        
        self.rgb = torch.from_numpy(np.load(BytesIO(data['eigen_test_rgb.npy']))).type(torch.float32) #Range [0,1]
        self.depth = torch.from_numpy(np.load(BytesIO(data['eigen_test_depth.npy']))).type(torch.float32) #Range[0, 10]

    def __getitem__(self, idx):
        image = self.rgb[idx]
        depth = self.depth[idx]
        return image, depth

    def __len__(self):
        return len(self.rgb)



def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    nyu2_test = list((row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))

    #Debugging
    #if True: nyu2_train = nyu2_train[:100]
    #if True: nyu2_test = nyu2_test[:100]

    print('Loaded (Train Images: {0}, Test Images: {1}).'.format(len(nyu2_train), len(nyu2_test)))
    return data, nyu2_train, nyu2_test


def train_transform(resolution):
    transform = transforms.Compose([
        Resize(resolution),
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor(test=False, maxDepth=10.0)
    ])
    return transform

def val_transform(resolution):
    transform = transforms.Compose([
        Resize(resolution),
        ToTensor(test=True, maxDepth=10.0)
    ])
    return transform


def get_NYU_dataset(zip_path, split, resolution='full', uncompressed=True):
    resolution = resolution_dict[resolution]
    if split == 'train':
        data, nyu2_train, nyu2_test = loadZipToMem(zip_path)

        transform = train_transform(resolution)
        dataset = depthDatasetMemory(data, split, nyu2_train, transform=transform)
    elif split == 'val':
        data, nyu2_train, nyu2_test = loadZipToMem(zip_path)

        transform = val_transform(resolution)
        dataset = depthDatasetMemory(data, split, nyu2_test, transform=transform)
    elif split == 'test':
        if uncompressed:
            dataset = NYU_Testset_Extracted(zip_path)
        else:
            dataset = NYU_Testset(zip_path)

    return dataset
