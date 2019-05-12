# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:01:37 2018

@author: Alfonso Sanchez
Udacity Data Science Nanodegree program
Deep Learning Project
"""
'''****************************************************************************
Importing Libraries
****************************************************************************'''
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
'''****************************************************************************
Defining the CNN Class
****************************************************************************'''

class CNN(nn.Module):
    
    #Class to create CNN, using as base a pretrained CNN. 
    #-input_size: Number of neurons in the input layer of the fully connected
    #             NN used as classifier.
    #-output-size: Number of neurons in the output layer of the fully connected
    #             NN used as classifier
    #-hidden layers: Number of neurons on each of the hidden layers used to 
    #                create the fully connected CNN, in a form of a numpy 
    #                array, where the first element reprsent the number of neu-
    #                ons in the first layers and similar to the rest of the hi-
    #                dden layers.
    #-drop_p: Dropout layers used for the fully connected neural network. It 
    #         needs to be a numpy array, where the firs element represent the
    #         dropout of the first layer and so on, and then the same for the 
    #         rest of the layers used. In case a layer dont have a dropout layer
    #         it should have a 0 element in the array.
    #-preload: The name of the preoloaded CNN used as base of the desire
    #          classifier. 
    #-class_to_idx: In case loading a trainned CNN, we can use these instance
    #               variable to define the label used for the training of the
    #               network.
    
    #initializing the network
    def __init__(self, input_size, output_size, hidden_layers, drop_p, preload='vgg16', class_to_idx=None):
   
        super().__init__()
        #Checking if the pretained vgg16 network was used
        if preload ==  'vgg16':
            self.model = models.vgg16(pretrained=True)
        elif preload ==  'vgg16_bn':
            self.model = models.vgg16_bn(pretrained=True)
        else:
            print('Unknown pretrained model')

        #Creating the structure of the desired fully connected NN
        OrderedDictb = []
        layers_size = [input_size]

        n=0
        while n< len(hidden_layers):
            layers_size.append(hidden_layers[n])
            OrderedDictb.append(('fc'+str(n+1),nn.Linear(layers_size[n], layers_size[n+1])) )
            OrderedDictb.append(('relu'+str(n+1), nn.ReLU()))
            if drop_p[n]!=0:
                OrderedDictb.append(('drop'+str(n+1), nn.Dropout(p=drop_p[n])))
            n+=1
        OrderedDictb.append(('fc'+str(n+1),nn.Linear(layers_size[n], output_size)))
        OrderedDictb.append(('output', nn.LogSoftmax(dim=1)))
        self.model.classifier = nn.Sequential(OrderedDict(OrderedDictb))
        
        self.model.class_to_idx = class_to_idx
        
    #Defining the forward method for the network
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in zip(self.model.classifier.hidden_layers):
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)