# Overall Imports
import pandas as pd
import matplotlib.pyplot as plt
from random import random
import numpy as np

#Torch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#Imports from other files 
from first_model import VanillaCNN
from third_model import AutoConvDeconv
from second_model import PoolUnpool

# warnings
import warnings
warnings.filterwarnings("ignore")

#############
# Utilitiess#
#############

# create a dictionnary that map classes to random pixels (RGB)
def create_dictionnary(nbr_classes):
    dict_pixels = {}
    for i in range(nbr_classes):
        dict_pixels[str(i)]= [int(random()*255),int(random()*255),int(random()*255)]
    return(dict_pixels)

# map the classes to the pixels of the dictionnary
def get_back_pixels(picture, dict_classes):

    picture_shape = list(picture.shape)
    new_picture = np.zeros(picture_shape[:-1]+[3], dtype=np.uint8)
    for i in range(picture_shape[0]):
        for j in range(picture_shape[1]):
            pixels = dict_classes[str(int(picture[i][j][0]))]
            for k in range(3):
                new_picture[i][j][k] = pixels[k]
    return(new_picture)

###################
#Display Functions#
###################

#display an image from an array
def show_image(before, title):
    axes = plt.subplots(nrows=1, ncols=2)[1]
    ax = axes.ravel()
    ax[0].imshow(before.astype(np.uint8))
    ax[0].set_title(title)      
    ax[0].axis('off')
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()

# get output and transform it into an array to display (with pixel restoration)
def show_output(network, data, classes_dictionnary, device):
    data = data.to(device)
    output = network(data)
    pred = output.data.max(1, keepdim=True)[1]
    pred = pred.cpu().detach().numpy()[0]
    pred = pred.transpose((1,2,0))
    out_picture = get_back_pixels(pred, classes_dictionnary)
    show_image(out_picture, 'Output')

# transform target into an array to display (with pixel restoration)
def show_target(target_tensor, classes_dictionnary):
    target = target_tensor.cpu().detach().numpy()[0]
    target = target.transpose((1,2,0))
    target_picture = get_back_pixels(target, classes_dictionnary)
    show_image(target_picture, 'Target')

# transform the tensor into an array to display
def show_picture(picture_tensor):
    picture_tensor = picture_tensor.detach().numpy()[0]
    picture = picture_tensor.transpose((1,2,0))
    picture = picture*255
    show_image(picture, 'Original picture')

# main function: test the trained network and display examples
def display_pictures_from_model(loader_path, model_path,device, classes_dictionnary, class_number, image_size, model_type,max_display=1, display_input=True,display_target=True, display_output=True):
    # loading models 
    network = 0
    if model_type == 'autoconvdeconv':
        network = AutoConvDeconv(class_number, image_size)
    if model_type == 'poolUnpool':
        network = PoolUnpool(class_number, image_size)
    if model_type == 'vanilla':
        network = VanillaCNN(class_number, image_size)
    network.load_state_dict(torch.load(model_path))
    network.to(device)
    loader = torch.load(loader_path)

    # displays
    with torch.no_grad():
        for display_number, (data, target) in enumerate(loader):
            print(data.shape)
            print(target.shape)
            if display_number>max_display-1:
                break
            else:
                if display_input:
                    show_picture(data)
                if display_target:
                    show_target(target, classes_dictionnary)
                if display_output:
                    show_output(network, data, classes_dictionnary, device)

# main of this file: test of the main function 
if __name__ == "__main__":
    class_count = 5
    image_size = [3,256,256]
    classes_dictionnary = create_dictionnary(class_count)
    display_pictures_from_model('./loaders/loader_autoConvdeconv_25_30_p.pth', './models/model__autoConvdeconv_25_30_p.pt', 'cuda:0',classes_dictionnary, class_count, image_size,'autoconvdeconv')
