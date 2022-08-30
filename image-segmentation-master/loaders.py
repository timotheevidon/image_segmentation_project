# Overall Imports
import numpy as np
import pandas as pd
import os 
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Torch Imports
from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms

#############
# Utilitiess#
#############

# loading inputs after transformation 
def load_input_from_folder(path):
	list_images = []
	for filename in os.listdir(path):
		list_images.append(plt.imread(path+'/'+filename)) 
	return(np.array(list_images))

# loading outputs after transformation
def load_output_from_folder(path):
	list_images = []
	for filename in os.listdir(path):
		list_images.append(pd.read_csv(path+'/'+filename)) 
	return(np.array(list_images))

# load inputs and outputs after transformation
def load_data(path_input, path_output, size_test):
	inputs = load_input_from_folder(path_input)
	outputs = load_output_from_folder(path_output)
	return(train_test_split(inputs, outputs, test_size=size_test, random_state=42))

# load data and return a dictionnary, key : filename, value: array of the outputs
def load_output_from_folder_to_dict(path, list_indexes):
	dict_images = {}
	for filename in os.listdir(path):
		if filename in list_indexes:
			dict_images[filename] = pd.read_csv(path+'/'+filename).values[:, 1:]
	return(dict_images)

################
# DataSet Class#
################

class CustomDataset(Dataset):
	#init function: attributes initialization
	def __init__(self, folder_inputs,folder_outputs,list_indexes, transform=None):
		self.list_indexes = [str(i)+'.csv' for i in list_indexes]
		self.landmarks_frame = load_output_from_folder_to_dict(folder_outputs, self.list_indexes)
		self.folder_inputs = folder_inputs
		self.folder_outputs = folder_outputs
		self.transform = transform

	# len function: return the length of the available data
	def __len__(self):
		return len(self.landmarks_frame)

	# getitem function: return the element when called with []
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()[0]
		idx_dataset = self.list_indexes[idx]
		image = plt.imread(self.folder_inputs+str(idx_dataset[:-4])+'.jpg')
		landmarks = self.landmarks_frame[str(idx_dataset)]
		landmarks = np.array([landmarks])
		landmarks = np.transpose(landmarks, (1,2,0))
		sample = {'image': image, 'landmarks': landmarks}
		if self.transform:
			sample['image'] = self.transform(sample['image'])
			sample['landmarks'] = self.transform(sample['landmarks'])
		return sample['image'], sample['landmarks']

######################
# Create Dataloaders #
######################

# split into train and test set of filenames
def train_test_split_filenames(path, split_size, small):
	list_filenames= [name[:-4] for name in os.listdir(path)]
	if small:
		list_filenames = list_filenames[0:100]
	random.shuffle(list_filenames)
	end_train = int(split_size[0]*len(list_filenames))
	return(list_filenames[0:end_train],list_filenames[end_train:])

# split into train, test and val set of filenames
def train_test_val_split_filenames(path, split_size, small):
	list_filenames= [name[:-4] for name in os.listdir(path)]
	if small:
		list_filenames = list_filenames[0:100]
	random.shuffle(list_filenames)
	end_train = int(split_size[0]*len(list_filenames))
	end_test = end_train + int(split_size[1]*len(list_filenames))
	return(list_filenames[0:end_train],list_filenames[end_train:end_test], list_filenames[end_test:])

# creation of three DataSets with train, test & val
# creation of three Dataloaders with the DataSets
def train_test_split_and_get_dataloaders(split_size,path_inputs, path_outputs, batch_size, val=False, small=True):
	data_transform = transforms.Compose([transforms.ToTensor()])
	if val:
		train_ids, test_ids, val_ids = train_test_val_split_filenames(path_inputs, split_size, small)
		val_set = CustomDataset(path_inputs, path_outputs,val_ids, transform = data_transform)
		val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
	else:
		train_ids, test_ids = train_test_split_filenames(path_inputs, split_size, small)
	train_set = CustomDataset(path_inputs, path_outputs,train_ids, transform = data_transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
	test_set = CustomDataset(path_inputs, path_outputs,test_ids, transform = data_transform)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
	
	# displays the sizes
	if val:
		print('train_loader_size: ', len(train_set))
		print('test_loader_size: ', len(test_set))
		print('val_loader_size: ', len(val_set))
		return(train_loader, test_loader, val_loader)
	else:
		print('train_loader_size: ', len(train_set))
		print('test_loader_size: ', len(test_set))
		return(train_loader, test_loader)