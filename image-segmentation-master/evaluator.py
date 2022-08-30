# Overall Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
#Torch Imports

import torch
import torch.optim as optim

#Imports from other files 
from loaders import train_test_split_and_get_dataloaders
from utils import train, test
from first_model import VanillaCNN
from second_model import PoolUnpool
from third_model import AutoConvDeconv
from tester import get_back_pixels, create_dictionnary, show_image


def grid_search(device, networks, optimizer_str, small, batch_sizes, epochs_list, lrs, path_input, path_output, size_split, num_classes, size_picture):
	'''
	Main function of the project: grid search on the different models and testing different batch size, learning rate and number of epochs
	args: 
		device : ['cpu', 'cuda']
		networks : [VanillaCNN, PoolUnpool, AutoConvDeconv]
		optimizer_str : name of the optimizer (adam or sdg)
		small : take only 100 pic sample for tests
		batch_sizes : list of the batch_sizes
		epochs_list : list of the numbers of epochs
		lrs : list of the numbers of lr
		path_input : path to get the pictures
		path_output : path to get the target
		size_split : % for each dataset
		val : [True, False] if val loader needed
		num_classes : number of classes in the dataset
		size_picture : shape of the picture

	return:
		report : a summary of all the models tested

	'''
	grid_search_report = [['name','epochs','batch_size','lr','score','loss']]
	best_network = 0
	best_score = 0
	best_val_loader = 0
	optimizer = 0
	for batch_size in batch_sizes:
		train_loader, test_loader, val_loader = train_test_split_and_get_dataloaders(size_split, path_input, path_output, batch_size, True, small)
		for epochs in epochs_list:
			for lr in lrs:
				for network in networks:
					network.to(device)
					print('Working on model : ',network.name, ' with batch_size :',batch_size, ' epochs : ', epochs, ' lr : ', lr, ' optimizer : ', optimizer_str)
					if optimizer_str == 'sgd':
						optimizer = optim.SGD(network.parameters(), lr=lr)
					if optimizer_str == 'adam':
						optimizer = optim.Adam(network.parameters(), lr=lr)
					#Training
					for epoch in tqdm(range(epochs)):
						train(network, train_loader, optimizer, device)
					#Testing
					score, loss = test(network, test_loader, optimizer, device, size_picture[2], True)
					if score > best_score:
						best_score = score
						best_val_loader = val_loader
						best_network = network
					#Reporting and saving
					grid_search_report.append([network.name, epochs, batch_size, lr, score.item(), loss.item()])
					report_df = pd.DataFrame(grid_search_report)
					report_df.to_csv('./reports/{}.csv'.format(report_name), index=False)
					torch.save(network.state_dict(), './models/model_'+'_'+str(network.name)+'_'+str(batch_size)+'_'+str(epochs)+'_p'+str(lr)[2:]+'.pt')
					torch.save(val_loader, './loaders/loader'+'_'+str(network.name)+'_'+str(batch_size)+'_'+str(epochs)+'_'+'p'+str(lr)[2:]+'.pth')
					del network 
	return best_network, best_val_loader, optimizer

# main of this file: test of the main function 
if __name__ == "__main__":
	torch.cuda.memory_summary(device=None, abbreviated=False)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	small = False # This allows us to get a fraction of the dataset for testing
	# The number of class is normally 22, but they are only 5 classes in the examples
	num_classes = 5
	size_picture = [3, 256, 256]
	size_split = [0.6, 0.2, 0.2]
	path_input = './data/inputs/'
	path_output ='./data/outputs'

	### PARAMS ###
	report_name = 'report example'
	networks = [AutoConvDeconv(num_classes, size_picture)]#, PoolUnpool(num_classes, size_picture), AutoConvDeconv(num_classes, size_picture)]
	optimizer_str = 'adam'
	batch_sizes = [25]
	epochs_list = [30]
	lrs = [0.1,1]
	best_network, best_val_loader, optimizer = grid_search(device, networks, optimizer_str, small, batch_sizes, epochs_list, lrs, path_input, path_output, size_split, num_classes, size_picture)

	### VALIDATION ###
	score, loss = test(best_network, best_val_loader, optimizer, device, size_picture[2], False)
	print('Validation score after grid_search : ', score.item(), ' woth loss : ', loss.item())