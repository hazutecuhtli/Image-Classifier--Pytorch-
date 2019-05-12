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
import sys, argparse, json
from PIL import Image
from torch import nn, optim
import torch.nn.functional as F
from Clase import CNN
from Functions import load_CNN, predict
'''****************************************************************************
Getting the parameters to run the script
****************************************************************************'''

parser = argparse.ArgumentParser()

parser.add_argument('image_path', help='Path for the image to classify')

parser.add_argument('checkpoint', help='File containing the trained CNN')

parser.add_argument('--top_k', action='store',
                    dest='topk',
                    help='Store the number of classes to aproximate the prediction')

parser.add_argument('--category_names ', action='store',
                    dest='catnames',
                    help='Category names for the prediction')

parser.add_argument('--gpu', action='store_true')

param = parser.parse_args()

paths = param.image_path
checkpoint = param.checkpoint
topk = param.topk
catnames = param.catnames
gpu = param.gpu

'''****************************************************************************
Defining the parameters used
****************************************************************************'''

if paths == None:
    print('The path for the folder where the data is located is required')

if checkpoint == None:
    print('The path for the folder where checkpoint is located is required')

if topk == None:
    topk = 1
else:
    topk = int(topk)

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
Loading the trained CNN
****************************************************************************'''

modelpre = load_CNN(checkpoint)

'''****************************************************************************
Classifying
****************************************************************************'''

psmax, prediction = predict(paths, modelpre.to(device), topk=topk, device=device)

'''****************************************************************************
Printing results
****************************************************************************'''

if catnames != None:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    predictionlab = [cat_to_name[x] for x in prediction]
    prediction = predictionlab
                         
if topk==1:
    print('The selected image was classified as class: ', str(prediction[0]))
    print('The probability of being class ' +str(prediction[0]) + ' is: ', str(psmax[0]))
else:
    predtemp = ''
    pstemp = ''
    for n in range(len(prediction)):
        if n != (len(prediction)-1):
            par = ', '
        else:
            par = ''
        predtemp = predtemp + str(prediction[n]) + par  
        pstemp = pstemp + str(psmax[n]) + par 
        
    print('The selected image can be classified as classes: ', predtemp)
    print('The probabilities of being classes ' + predtemp + ' are: ', pstemp, ', respectively.' )      

'''****************************************************************************
Fin
****************************************************************************'''