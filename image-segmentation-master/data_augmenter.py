# Overall Imports
import numpy as np
import os 
import random
import matplotlib.pyplot as plt
import pandas as pd

#skimage imports
from skimage import data
from skimage.transform import rescale
from skimage.util import random_noise
from skimage import exposure

#Torch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#Imports from other files 
from scipy import ndimage
from tester import show_image
import matplotlib.image as saving

# augmentation of the pictures, returns 5 pic per input
def data_augmentation(original_image, rescale=False, new_size=[0,0]):
  list_augmentation = []
  #original if rgb and grey version if not
  list_augmentation.append(original_image)
  #random noise
  list_augmentation.append(random_noise(original_image))
  #horizontal flip
  list_augmentation.append(original_image[:, ::-1])
  #blured
  list_augmentation.append(ndimage.uniform_filter(original_image, size=(11, 11, 1)))
  #contrast 
  v_min, v_max = np.percentile(original_image, (0.2, 99.8))
  list_augmentation.append(exposure.rescale_intensity(original_image, in_range=(v_min, v_max)))
  return(list_augmentation)

# take a bigger array in order to have the same size in width & height
def make_square(array_, size):
    array_size = array_.shape
    array_output = np.zeros((size[0],size[1], 3))
    array_output[0:array_size[0], 0:array_size[1], 0:3] = array_[0:array_size[0], 0:array_size[1], 0:3]
    return(array_output)

# resize is impossible due to the necessity of keeping the target unchanged 
# resize would mix the classes. That's why we crop
def crop_and_make_square(image, max_size,  size, fill_color=(0, 0, 0, 0)):
    size_image = image.shape
    if size_image[0]>=max_size[0]:
      if size_image[1]>max_size[1]:
        test = image[0:max_size[0],0:max_size[1], :]
        return(make_square(test, size))
      cropped = image[0:max_size[0], :, :]
      return(make_square(cropped, size))
    elif size_image[1]>=max_size[1]:
        cropped = image[:,0:max_size[1],:]
        return(make_square(cropped, size))
    else:
        return(make_square(image, size))

# transform the output from RGB to the class number
def transform_output(output_image, dict_classes):
  image_shape = output_image.shape
  array_output = np.zeros((image_shape[0], image_shape[1]))
  for i in range(image_shape[0]):
    for j in range(image_shape[1]):
      list_of_pixels = str(output_image[i][j])
      dict_keys = list(dict_classes.keys())
      if list_of_pixels not in dict_keys: 
        dict_classes[list_of_pixels] = len(dict_keys)
      array_output[i][j]= dict_classes[list_of_pixels]
  return(array_output, dict_classes)

#Main function: Take all the dataset and transform it before saving into the right format
def transform_data(path_input_output, path_input_input,size_picture, size_square, path_output= './data', verbose= False):
  try: 
      os.mkdir(path_output)
  except:
      pass
  try: 
      os.mkdir('./'+path_output+'/inputs')
  except:
      pass
  new_data = list()
  outputs_list = list()
  dict_classes = {} 
  list_filenames = os.listdir(path_input_output)
  j=0
  for filename in list_filenames:
      output_image =  plt.imread(path_input_output+'/'+filename)
      output_image = crop_and_make_square(output_image, size_picture, size_square)
      output_image, dict_classes= transform_output(output_image, dict_classes)
      try:
        original_image = plt.imread(path_input_input+'/'+filename[:-3]+'jpg')
        original_image = crop_and_make_square(original_image, size_picture, size_square).astype(np.uint8)
      except:
        print('exception, input not present in folder')
        continue
      for i in range(5):
          outputs_list.append(output_image)
      new_data += data_augmentation(original_image)
      if verbose:
        show_images(new_data[-5] , new_data[-random.randint(1,6)], 'test')
        show_images(new_data[-5] , output_image, 'test')
      j+=1
      print('- - - - - - - - - - - - - - -')
      print(filename+': ' + str(round(j/len(list_filenames)*100, 2))+ '% of pictures processing done')
  print('##################################')
  print("number of classes:", len(dict_classes))
  print('##################################')
  #saving
  print('___________ SAVING DATA ____________')
  for i in range(len(new_data)):
      #input
      saving.imsave(path_output+'/inputs/'+str(i)+'.jpg', new_data[i])
      #output
      pd.DataFrame(outputs_list[i]).to_csv(path_output+'/outputs/'+str(i)+'.csv')
  return(dict_classes)

# main of this file: preprocesses our dataset from examples to data
if __name__ == "__main__":
      transform_data('examples/targets/', 'examples/pictures/', [256, 256], [256, 256] )