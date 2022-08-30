# Imports
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

# softmax function in order to transform scores into probabilities
def softmax_custom(x):
    x = torch.transpose(x, 1, 2)
    x = torch.transpose(x, 2, 3)
    x= F.softmax(x, dim=3)
    x = torch.transpose(x, 3, 2)
    x = torch.transpose(x, 2, 1)
    return(x)

# loss function for the Neural Network
def get_loss(output, target):
    # in order to compare, we need the same shape
    output_int = torch.transpose(output, 0, 1)
    output_int = torch.flatten(output_int, start_dim=1)
    output_int = torch.transpose(output_int, 0, 1)
    target_int = torch.flatten(target, start_dim=0)
    # we use cross entropy on every pixel and the 22 output classes
    loss = nn.CrossEntropyLoss()
    return (loss(output_int.float(), target_int.long()))

# train function for the Neural Network
def train(network, loader, optimizer, device):
    #at each epoch we will use a loader to get batch and send
    # them throw the network
    network.to(device).train()
    for batch_idx, (data, target) in enumerate(loader):
        # to device is used to convert into Cuda if GPU available
        data = data.to(device)
        target = target.to(device)
        # optimizer to update the weights of the network
        optimizer.zero_grad()
        output = network(data)
        loss = get_loss(output, target)
        loss.backward()
        optimizer.step()
    
# test function for the Neural Network 
def test(network, loader, optimizer, device, size, verbose):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            # to device is used to convert into Cuda if GPU available
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += get_loss(output, target)
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1] 
            # sum of all pixels which are correctly estimated
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(loader.dataset)
    #print and return results
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, size*size*len(loader.dataset),
        100. * correct / (size*size*len(loader.dataset))))
    return(100. * correct / (size*size*len(loader.dataset)), test_loss)

