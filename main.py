#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:36:14 2021

@author: hienpham
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import os
import cv2

from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from glob import glob

class IDAO_dataset(Dataset):
    def __init__(self, path_dir, train=True):
        """
        Args:
            path_dir: path to dataset folder */idao_dataset/train
            train(bool): train/test set 
        """
        self.path_dir = path_dir
        self.train = train
        self.data = []
        
        if train:
            self.energy_class = {1:'NR', 3:'ER', 6:'NR', 10:'ER', 20:'NR', 30:'ER'}
        else:
            self.energy_class = {1:'ER', 3:'NR', 6:'ER', 10:'NR', 20:'ER', 30:'NR'}
        
        def slice_string(my_string, slice_prefix, slice_suffix):
            index_pre = my_string.find(slice_prefix)
            index_suf = my_string.find(slice_suffix)
            if (index_pre != -1) and (index_suf != -1):
                return int(my_string[index_pre+3:index_suf-1])
            else:
                raise Exception("Substring not found")
        
        #path_dir = /home/hienpham/Bureau/IDAO/Dataset/idao_dataset/train/
        for recoil_event in os.listdir(self.path_dir):
            sub_path = os.path.join(self.path_dir, recoil_event)
            for img_path in glob(sub_path + "/*.png"):
                energy_level = slice_string(file, recoil_event + "_", "keV")
                if self.energy_class[energy_level] == recoil_event:
                    self.data.append([img_path, energy_level, recoil_event])
        
        
    def __getitem__(self, idx):
        img_path, energy_level, recoi_event = self.data[idx]
        img = cv2.imread(img_path)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        energy_level_tensor = torch.tensor(energy_level)
        return img_tensor, energy_level_tensor

    def __len__(self):
        return len(self.data)
        
        
class idao_model(nn.Module):
    # use VGG for dummy training
    def __init__(self, features, output_dim):
        super().__init__()
        
        self.features = features
        
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h


def train (model):

    
path_dir = r'~/idao_dataset/train/'
train_set = IDAO_dataset(path_dir, train=True)
test_set = IDAO_dataset(path_dir, train=False)
# 6 classes of energy level
output_dim = 6