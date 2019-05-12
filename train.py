# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:44:13 2018

@author: Alfonso Sanchez
Udacity Data Science Nanodegree program
Deep Learning Project
"""
'''****************************************************************************
Importing Libraries
****************************************************************************'''
import os, sys, argparse
from torch import nn, optim
import torch.nn.functional as F
from Clase import CNN
from Functions import do_deep_learning, Processing_Datasets, savingCNN
'''****************************************************************************
Getting the parameters to run the script, from the user
****************************************************************************'''

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', help='Data directory')

parser.add_argument('-s', action='store',
                    dest='save_dir',
                    help='Saving directory')

parser.add_argument('--arch', action='store',
                    dest='architecture',
                    help='Achitecture for the CNN')

parser.add_argument('--learning_rate', action='store',
                    dest='eta',
                    help='Learning rate to train the CNN')

parser.add_argument('--hidden_units','--list', nargs='+', dest='hiddenlayers', help='Hidden layers for the CNN')

parser.add_argument('--epochs', action='store',
                    dest='epochs',
                    help='Epochs for the training of the CNN')

parser.add_argument('--gpu', action='store_true', help="Type of processing, 'cpu' or 'cuda'")

param = parser.parse_args()

paths = param.data_dir
hidden = param.hiddenlayers
epochs = param.epochs
architecture = param.architecture
eta = param.eta
gpu = param.gpu
savefile = param.save_dir

'''****************************************************************************
Defining the parameters to be used
****************************************************************************'''

if paths == None:
    print('The path for the folder where the data is located is needed')
if hidden == None:
    hidden = [1024, 512]
else:
    hidden = [int(x) for x in param.hiddenlayers]
if epochs==None:
    epochs = 10
else:
    epochs = int(epochs)
if architecture==None:
   architecture= 'vgg16'
if eta==None:
    eta = 0.01 
else:
    eta = float(eta)
if gpu == False:
    device = 'cpu'
else:
    device = 'cuda'
    
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
batchs = [64, 32, 32]
resizes=256
crops = 224
print_every = 102

'''****************************************************************************
Defining the training, testing and validation datasets
****************************************************************************'''

loaders, datasets = Processing_Datasets(paths, means, stds, batchs, resizes=resizes, crops=crops)

'''****************************************************************************
Creating the CNN
****************************************************************************'''

CNNr = CNN(25088, 102, hidden, [.5,.2], architecture, None)
CNNr = CNNr.model

'''****************************************************************************
Training the CNN
****************************************************************************'''

criterion = nn.NLLLoss()
optimizer = optim.SGD(CNNr.classifier.parameters(), lr = eta, momentum=0.9)

do_deep_learning(CNNr, loaders[0], epochs, criterion, optimizer, device=device)

'''****************************************************************************
Saving the trained CNN
****************************************************************************'''

if savefile == None:
    savingCNN('checkpoint.pth', CNNr, 25088, 102, hidden, [.5,.2], preload = architecture, class2idx=datasets[0].class_to_idx)
else:
    savingCNN(savefile, CNNr, 25088, 102, hidden, [.5,.2], preload = architecture, class2idx=datasets[0].class_to_idx)

'''****************************************************************************
Fin
****************************************************************************'''
