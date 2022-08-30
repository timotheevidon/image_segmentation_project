# Overall Imports
import os
import numpy as np
from tqdm import tqdm

#Torch Imports
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
from torch.autograd import Variable

#Imports from other files 
from loaders import train_test_split_and_get_dataloaders

# block the update of the layers
def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

#download resnet model and prepare it for fine tuning
def get_resnet(num_classes, device):
    model_ft = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model_ft)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_ft = model_ft.to(device)
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    return(model_ft, params_to_update)

#train function adapted to fine tuning
def train(network, loader, optimizer, device):
    network.to(device).train()
    loss_ = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        target = torch.flatten(target, start_dim=1)
        print(output.shape)
        print(target.shape)
        loss = loss_(output.float(), target.long())#.item()
        loss.backward()
        optimizer.step()

#test function adapted to fine tuning
def test(network, loader, optimizer, device, size):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += nn.CrossEntropyLoss(output, target)#.item()
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, size*size*len(loader.dataset),
        100. * correct / (size*size*len(loader.dataset))))
    return(100. * correct / (size*size*len(loader.dataset)), test_loss)

#return the loss
def get_loss(output, target):
    output_int = torch.transpose(output, 0, 1)
    output_int = torch.flatten(output_int, start_dim=1)
    output_int = torch.transpose(output_int, 0, 1)
    target_int = torch.flatten(target, start_dim=0)
    loss = nn.CrossEntropyLoss()
    return (loss(output_int.float(), target_int.long()))

# main of this file: test of the main function 
# warning: not working
if __name__ == "__main__":
    num_classes = 256*256*22
    batch_size = 25
    epochs = 10
    lr = 0.1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #network, params_to_update  = get_resnet(num_classes, device)
    network = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=22, pretrained=True)
    params_to_update = network.parameters
    print('done model')
    optimizer_ft = optim.SGD(params_to_update, lr)
    print('opti')
    train_loader, test_loader, val_loader = train_test_split_and_get_dataloaders([0.6, 0.2, 0.2],'./data/inputs/','./data/outputs',batch_size, val=True, small = True )
    print('loader done')
    for epoch in tqdm(range(epochs)):
        train(network, train_loader, optimizer_ft, device)
    test(network, test_loader, optimizer_ft, device, 256)
    