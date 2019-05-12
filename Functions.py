# -*- coding: utf-8 -*-
"""
Editor de Spyder

@author: Alfonso Sanchez
Udacity Data Science Nanodegree program
Deep Learning Project
"""
'''****************************************************************************
Importing Libraries
****************************************************************************'''
import matplotlib.pyplot as plt
import torch, random, time
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from Clase import CNN
from PIL import Image
'''****************************************************************************
Defining the train, test and validation data
****************************************************************************'''

def Processing_Datasets(paths, means, stds, batchs, resizes=256, crops=224):
        
    #Prepare data to be used for the training, testins and validation
    #of the a desired classifier, where:
    #-path: is the path where the data is located, which needs to contains
    #       specific folders for the training, testins and validation data.
    #-means: To be use to normalize the data by substracting them.
    #-stds: To be used for normalization.
    #-Resize: Size of the data to be processed
    #-Crop: Desired size of the data to be used in the classifier.
    #-batchs: batch sizes for the training, testing and validation datasets,
    #         respectively. They need to be in a numpy array form. 
    
    #Preparing the directories where the data is located:
    data_dir = paths
    train_dir = paths + '/train'
    valid_dir = paths + '/valid'
    test_dir = paths + '/test'
    
    #Defining the transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.Resize((256,256)),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize((resizes,resizes)),
                                          transforms.CenterCrop(crops),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means,stds)])
    
    validation_transforms = transforms.Compose([transforms.Resize((resizes,resizes)),
                                          transforms.CenterCrop(crops),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means,stds)])
    
    #Loading the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    
    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batchs[0], shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batchs[1])
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=batchs[2])
    
    Datasets = [train_data, test_data, validation_data]
    loaders = [trainloader, testloader, validationloader]
    
    return loaders, Datasets
    
'''****************************************************************************
Loading CNN 
****************************************************************************'''

def load_CNN(filepath):
    
    #Loads a saved CNN, where:
    #-filepath: is the location where the CNN was saved, or is located. 
    
    #Loading the desired CNN
    checkpoint = torch.load(filepath)
    premodel = CNN(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'],
                             checkpoint['drops_p'],
                             checkpoint['preload'],
                             checkpoint['class_to_idx'])

    premodel.model.load_state_dict(checkpoint['state_dict'])
    
    return premodel.model

'''****************************************************************************
Training the CNN
****************************************************************************'''

def do_deep_learning(model, trainloader, epochs, criterion, optimizer, device='cpu'):
    
    #Training a CNN to be used as classifier to process images, where:
    #-model: is the CNN netwotk architecture to train
    #-trainloder: the loader that contains the training data
    #-epochs: the number of epochs to be used for the training of the CNN
    #-print_every: Prin the status of the training CNN every specic steps
    #-criterion: The criterioum used to train the network
    #-optimizer: The optimizer used to train the network
    #-device: Training the networks using the 'cpu' or 'cuda' (GPU)
    
    steps = 0
    
    # changing to cuda (GPU)
    model.to('cuda')
    
    train_losses = []
    
    #Training the network
    start = time.time()
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            
            images, labels = images.to('cuda'), labels.to('cuda')
    
            optimizer.zero_grad()
            
            # Forward and backward passes
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() 
        else:
            accuracy = 0
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for images, labels in trainloader:
                    
                    images, labels = images.to('cuda'), labels.to('cuda')
                    
                    log_ps = model(images)
                   
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                      
            train_losses.append(running_loss/len(trainloader))
    
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Training Accuracy: {:.3f}.. ".format(accuracy/len(trainloader)),
                  "Time{:.2f}:".format((time.time() - start)))
            start = time.time()
               
            running_loss = 0
                
'''****************************************************************************
saving a CNN
****************************************************************************'''

def savingCNN(filename, model, inputsize, outputsize, hiddenlayers, dropoutlayers, preload = None, class2idx=None):
    
    #Saving a trained CNN, where:
    #-model: The trained CNN to be saved
    #-inputsize: The number of units in the input of the CNN
    #-outputsize: The number of units in the output layer
    #-hiddenlayers: Number of units on each of the hidden layers used, list.
    #-dropoutlayers: The numner of dropout layers to be used as a list, where
    #the probability of each layer is defined in the list. If not layer is used
    #for a specic stage of the CNN a probability o 0 should be used.
    #-preload: Preoladed model to be used
    #-class2idx: Labels used for the batches of the trained CNN    
    
    checkpoint = {'input_size': inputsize,
                  'output_size': outputsize,
                  'hidden_layers': hiddenlayers,
                  'drops_p': dropoutlayers, 
                  'preload': preload,
                  'class_to_idx': class2idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, filename)
    
'''****************************************************************************
Processing images
****************************************************************************'''

def process_image(image):

    #Processing an image to be used in a trained CNN, where:
    #-image:The path of the image to be processed
    
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    image = image.transpose(Image.ROTATE_90)    
    image = image.resize((256,256))
    image = image.crop((16,16,240,240))
    np_image = np.array(image)
    np_image = np_image/255
    np_image[:,:,0] = (np_image[:,:,0] - 0.485)/0.229
    np_image[:,:,1] = (np_image[:,:,1] - 0.456)/0.224
    np_image[:,:,2] = (np_image[:,:,2] - 0.406)/0.225
    np_image = np_image.T
    imagen = torch.from_numpy(np_image)
    
    return imagen

'''****************************************************************************
Predicting images
****************************************************************************'''

def predict(image_path, model, topk=5, device='cuda'):

    #Using a trained CNN to predict or classify images, where
    #-model: is the CNN netwotk architecture to be used 
    #-topk: Number of classes to consider for the prediction
    #-device: Type of processing used for the prediction, 'cpu' or 'cuda' (GPU)
    
    #Implementing the code to predict the class from an image file
    image = img = Image.open(image_path)
    imagen = process_image(image)
    imagen = imagen[None, :, :, :]   
    imagen = imagen.float()
    
    with torch.no_grad():
        outputs = model(imagen.to(device))
    ps = F.softmax(outputs, dim=1)
    psmax = ps.topk(topk)
    psmax, psind = psmax[0][0].to('cpu').numpy(), psmax[1][0].to('cpu').numpy()

    prediction = []
    idx_to_class = {str(v): k for k, v in model.class_to_idx.items()}
    for indice in psind:
        prediction.append(idx_to_class[str(indice)])
    
    return psmax, prediction

'''****************************************************************************
Fin
****************************************************************************'''