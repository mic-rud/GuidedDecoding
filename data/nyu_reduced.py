import csv
import os
import random
from io import BytesIO
from pathlib import Path
from random import shuffle
from typing import List, Tuple
from zipfile import ZipFile

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import Resize, RandomHorizontalFlip, RandomChannelSwap, ToTensor
from config import SEED

random.seed(SEED)
torch.manual_seed(SEED)

resolution_dict = {
    'full' : (480, 640),
    'half' : (240, 320),
    'mini' : (224, 224)}

class depthDatasetMemory(Dataset):
    def __init__(self, data: List[Tuple[str, str]], split, transform=None):
        self.data = data
        self.transform = transform
        self.split = split

    def __getitem__(self, idx):
        sample = self.data[idx]
        base_path = '/home/x/Área de Trabalho/GuidedDecoding/data'
        image = Image.open(Path(base_path, sample[0]))
        depth = Image.open(Path(base_path, sample[1]))
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
        return len(self.data)

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
        self.data = data
        #self.rgb = torch.from_numpy(np.load(BytesIO(data['eigen_test_rgb.npy']))).type(torch.float32) #Range [0,1]
        #self.depth = torch.from_numpy(np.load(BytesIO(data['eigen_test_depth.npy']))).type(torch.float32) #Range[0, 10]

    def __getitem__(self, idx):
        #image = self.rgb[idx]
        #depth = self.depth[idx]

        # TODO: Editar para deixar de ser hard coded
        data_numpy = np.load(BytesIO(self.data['nyu_test_001.npz']))
        image = torch.from_numpy(data_numpy['image']).type(torch.float32)
        depth = torch.from_numpy(data_numpy['depth']).type(torch.float32)
        return image, depth

    def __len__(self):
        return len(self.data)



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


def get_NYU_dataset(zip_path, split, resolution='full', uncompressed=False):
    resolution = resolution_dict[resolution]

    if split != 'test':
        data = read_nyu_csv(zip_path)
        shuffle(data)

        if split == 'train':
            data_train = data
            transform = train_transform(resolution)
            dataset = depthDatasetMemory(data_train, split, transform=transform)

        elif split == 'val':
            data_val = data
            transform = val_transform(resolution)
            dataset = depthDatasetMemory(data_val, split, transform=transform)

    elif split == 'test':
        if uncompressed:
            dataset = NYU_Testset_Extracted(zip_path)
        else:
            dataset = NYU_Testset(zip_path)

    return dataset


def read_nyu_csv(csv_file_path) -> List[Tuple[str, str]]:
    """
    Lê CSV que relacionada x e y e retona uma lista de pares de paths (x, y)
    :param csv_file_path: Path do arquivo CSV com o nome de x e y
    :return: Lista de pares (path input, path ground truth)
    """
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        return [('./' + row[0], './' + row[1]) for row in csv_reader if len(row) > 0]
