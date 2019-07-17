import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline
plt.ion()   # interactive mode

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader


from CustomModel import PretrainedDensenet
from read_data import Data
from MuraDatasets import MuraDataset
from train import train
from CustomLoss import Loss
from visualize import see_samples, view_data_count

device = torch.device("cpu")


# '''
parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--use_cuda', type=bool, default=False, help='device to train on')
parser.add_argument('--samples', type=bool, default=False, help='See sample images')
parser.add_argument('--view_data_counts', type=bool, default=False, help='Visualize data distribution')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train on')
parser.add_argument('--train', default=True, type=bool, help='train the model')

opt = parser.parse_args()


if opt.use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# '''


df = Data()

train_df = df.train_df
valid_df = df.valid_df
train_labels_data = df.train_labels_data
valid_labels_data = df.valid_labels_data

if opt.samples:
    see_samples(train_df)
# plt.show()

train_df['Label'] = train_df.apply(lambda x:1 if 'positive' in x.FilePath else 0, axis=1)
train_df['BodyPart'] = train_df.apply(lambda x: x.FilePath.split('/')[2][3:],axis=1)
train_df['StudyType'] = train_df.apply(lambda x: x.FilePath.split('/')[4][:6],axis=1)
valid_df['Label'] = valid_df.apply(lambda x:1 if 'positive' in x.FilePath else 0, axis=1)
valid_df['BodyPart'] = valid_df.apply(lambda x: x.FilePath.split('/')[2][3:],axis=1)
valid_df['StudyType'] = valid_df.apply(lambda x: x.FilePath.split('/')[4][:6],axis=1)
train_df.set_index(["FilePath", "BodyPart"]).count(level="BodyPart")

train_df.set_index(["FilePath", "Label"]).count(level="Label")

if opt.view_data_counts:
    view_data_count(train_df, valid_df)


if opt.train:
    # The paper uses the same standard deviation and mean as that of IMAGENET dataset
    train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.456], [0.224])
        ])
    val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.456], [0.224])
        ])

    train_dataset = MuraDataset(df=train_df,transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, num_workers=0, shuffle=True)
    val_dataset = MuraDataset(df=valid_df, transform=val_transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, num_workers=0, shuffle=True)


    model = PretrainedDensenet()
    model.to(device)

    criterion = Loss(train_df, valid_df, device)

    # The network was trained end-to-end using Adam with default parameters β1 = 0.9 and β2 = 0.999 
    optimizer = optim.Adam(model.parameters(), betas=(0.9,0.999), lr=0.0001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

    model_ft = train(criterion=criterion,optimizer=optimizer, model=model, n_epochs=opt.n_epochs, device=device, train_loader=train_loader, val_loader=val_loader)
    