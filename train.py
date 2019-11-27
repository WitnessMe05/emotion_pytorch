# import modules!
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gzip, pickle
import numpy as np
import pandas as pd
import subprocess
import os
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter

#
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        elif os.path.exists(directory):
            print("Directory is already exists")
    except OSError:
        print ('Error: Creating directory. ' +  directory)


classes = 4
speak = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05'] # cross validation

for sp in speak:
    class FeatureData(Dataset):

        def __init__(self, is_train_set=False, transforms=None):
            le = preprocessing.LabelEncoder()
            filename = "./test/IS09_emo.csv"
            with gzip.open("/home/gnlenfn/remote/pytorch_emotion/IS09_feature.pkl") as f:
                feat = pickle.load(f)
            if is_train_set == False:
                data = feat[feat['Session'] == sp]
            else:
                data = feat[feat['Session'] != sp]

            # Pick a record type
            data = data[data['Type'] == 'impro']
            
            # Emotion classes
            neu = data[data.EMOTION == 'neu']
            ang = data[data.EMOTION == 'ang']
            sad = data[data.EMOTION == 'sad']
            hap = data[data.EMOTION == 'hap']
            exc = data[data.EMOTION == 'exc']
            hap_m = pd.concat([hap, exc]).replace("exc", "hap").reset_index(drop=True) # Merge happy and excited

            self.emo_sep = pd.concat([neu, hap_m, sad, ang]).reset_index(drop=True)
            self.len = self.emo_sep.shape[0]
            self.x_data = torch.tensor(self.emo_sep['feature'])
            self.y_data = self.emo_sep['label']
            self.y_data = le.fit_transform(self.y_data)
            self.y_data = torch.tensor(self.y_data)

        def __getitem__(self, idx):
            return self.x_data[idx], self.y_data[idx]

        def __len__(self):
            return self.len






 