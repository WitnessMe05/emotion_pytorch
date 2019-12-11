# import modules!
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

import gzip, pickle
import numpy as np
import pandas as pd
import subprocess
import os
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
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


classes = 2
emo = ['ang', 'neu']
speak = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05'] # cross validation
avg_acc = 0

for sp in speak:

    # FEATURE DATASET CLASS
    class FeatureData(Dataset):

        def __init__(self, is_train_set=False, transforms=None):
            le = preprocessing.LabelEncoder()
            #filename = "./test/IS09_emo.csv"
            with gzip.open("/home/gnlenfn/remote/pytorch_emotion/Feature/IS10_paraling/IS10_paraling.pkl") as f:
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

            emo_sep = pd.concat([neu, ang]).reset_index(drop=True)
            self.len = emo_sep.shape[0]
            tmpdf = emo_sep.iloc[:,1:-3]
            #print(tmpdf.head())
            self.x_data = torch.tensor(tmpdf.values)
            self.y_data = emo_sep['EMOTION']
            self.y_data = le.fit_transform(self.y_data)
            self.y_data = torch.tensor(self.y_data)

        def __getitem__(self, idx):
            return self.x_data[idx], self.y_data[idx]

        def __len__(self):
            return self.len

    # MODEL CLASS
    class CNN(torch.nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.keep_prob = 0.5

            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2, stride=2)
            )

            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv1d(16, 24, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=1 - self.keep_prob)
            )

            self.layer3 = torch.nn.Sequential(
                torch.nn.Conv1d(24, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2, stride=2)
            )

            self.layer4 = torch.nn.Sequential(
                torch.nn.Conv1d(32, 40, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=1 - self.keep_prob)
            )

            self.layer5 = torch.nn.Sequential(
                torch.nn.Conv1d(40, 48, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=1 - self.keep_prob)
            )

            self.linear1 = torch.nn.Linear(2352, 64, bias=True)
            torch.nn.init.xavier_uniform_(self.linear1.weight)

            self.dense = torch.nn.Sequential(
                self.linear1,
                torch.nn.ReLU(),
                torch.nn.Dropout(p=1 - self.keep_prob)
            )

            self.linear2 = torch.nn.Linear(64, 4, bias=True)
            torch.nn.init.xavier_uniform_(self.linear2.weight)
            
        def forward(self, x):
            #print(x.shape)
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            #print(out.shape)
            out = out.view(out.size(0), -1)
            out = self.dense(out)
            out = self.linear2(out)
            return out

    # HYPERPARAMETERS
    learning_rate = 1e-3
    epochs = 5000
    batch_size = 1024

    # Seed for reproducibility
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed(777)
    torch.cuda.set_device(3)

    # DATA LOADER
    trans = transforms.Compose([transforms.ToTensor()])
    print("Data Loading...")
    train_dataset = FeatureData(is_train_set=True, transforms=trans)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset = FeatureData(is_train_set=False, transforms=trans)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, num_workers=0, drop_last=False)
    print("{} dataset on training...\n".format(sp))

    model = CNN().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
########################################################################################################################
    # TRAINING
    total_batch = len(train_loader)
    test_batch = len(test_loader)

    model.train()
    model_path = "./Model/2emo/"+sp+"/"
    print(sp + "Learning started. It takes some time...\n")
    for epoch in range(epochs):
        avg_cost = 0

        for X, Y in train_loader:
            X = X.unsqueeze(1).float().to(device)
            Y = Y.long().to(device)

            hypothesis = model(X)
            cost = criterion(hypothesis, Y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch

        if (epoch + 1) % 100 == 0:
            print('[Epoch: {:>4}] cost = {:>.10}, lr = {}'.format(epoch + 1, avg_cost, optimizer.param_groups[0]['lr']))
        
    # # SAVE MODEL
    createFolder(model_path)
    model_name = model_path + sp + "_" + str(epochs) + '_IS10'
    torch.save(model.state_dict(), model_name + ".pth")
    print("Model saved...")

    trained_model = CNN().to(device)
    trained_model.load_state_dict(torch.load(model_name + ".pth"))
    print("Model Loaded!")

    # TEST WITH TEST DATA
    acc = []
    heat = np.zeros([classes, classes], dtype=np.int)

    with torch.no_grad():
        model.eval()
        for X, Y in test_loader:
            X = X.unsqueeze(1).float().to(device)
            Y = Y.long().to(device)

            prediction = trained_model(X)
            correct_pred = torch.argmax(prediction, 1) == Y

            acc.append(correct_pred.float().mean())
            matrix = confusion_matrix(Y.cpu(), torch.argmax(prediction, 1).cpu(), labels=[x for x in range(classes)])
            heat += matrix

    accuracy = torch.tensor(acc).mean()

    # PRINT RESULT
    print("Accuracy: {}".format(accuracy.item()))
    avg_acc += accuracy.item()

    # CONFUSION MATRIX
    print(heat)
    dd = pd.DataFrame(heat, index=emo, columns=emo)
    k = dd.sum(axis=1)
    dd = dd.div(k, axis=0)

    plt.figure(figsize=(10,7))
    plt.title("Confusion Matrix of " + sp + "Test Set")
    conf1 = sns.heatmap(dd, annot=True, fmt=".0%", cmap='YlGnBu', cbar=False, vmin=0, vmax=1)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    cbar = conf1.figure.colorbar(conf1.collections[0])
    cbar.set_ticks([0,1])
    cbar.set_ticklabels(["0%", "100%"])
    plt.savefig(model_name + ".png")
    plt.show()

    print("###############"+sp+" ended###############\n\n")

# RESULT OF 5 FOLD
summary(model, input_size=(1,384))
print("\nAverage of 5-fold: ", avg_acc/5)




