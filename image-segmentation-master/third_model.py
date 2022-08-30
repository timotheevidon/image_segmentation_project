# General Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# imports from others files
from utils import softmax_custom

# Encoder-Decoder and Auto-Encoder Models: the tenet of the algorithm is to downscale the initial image into small 
# dimension and afterwards to upscale it back. This methodology aims to reconstruct the image with the segmentation.
class AutoConvDeconv(nn.Module):
    def __init__(self, class_number, image_size):
        super(AutoConvDeconv,self).__init__()
        self.name = 'autoConvdeconv'
        self.image_size = image_size
        self.class_number = class_number

        #Block 1
        self.conv_1_a = nn.Conv2d(in_channels=self.image_size[0], out_channels= 4, kernel_size= 3,padding=1)  
        self.conv_1_b = nn.Conv2d(in_channels=4, out_channels= 4, kernel_size= 3,padding=1)  
        self.max_pooling_1 = nn.MaxPool2d(2,2, return_indices=True)

        #Block 2
        self.conv_2_a = nn.Conv2d(in_channels=4, out_channels= 8, kernel_size= 3,padding=2)  
        self.conv_2_b = nn.Conv2d(in_channels=8, out_channels= 16, kernel_size= 5,padding=1) 
        self.max_pooling_2 = nn.MaxPool2d(2,2, return_indices=True)

        #Block 3 
        self.conv_3_a = nn.Conv2d(in_channels=16, out_channels= 16, kernel_size= 5,padding=3)  
        self.conv_3_b = nn.Conv2d(in_channels=16, out_channels= 16, kernel_size= 7,padding=3) 
        self.conv_3_c = nn.Conv2d(in_channels=16, out_channels= 32, kernel_size= 9,padding=3) 
        self.max_pooling_3 = nn.MaxPool2d(2,2, return_indices=True)

        #Block 4
        self.conv_4_a = nn.Conv2d(in_channels=32, out_channels= 32, kernel_size= 3,padding=2)  
        self.conv_4_b = nn.Conv2d(in_channels=32, out_channels= 32, kernel_size= 5,padding=2) 
        self.conv_4_c = nn.Conv2d(in_channels=32, out_channels= 64, kernel_size= 7,padding=2)
        self.max_pooling_4 = nn.MaxPool2d(2,2, return_indices=True)

        #Block 5
        self.conv_5_a = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size= 2,padding=1)
        self.conv_5_b = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size= 3,padding=1) 
        self.conv_5_c = nn.Conv2d(in_channels=64, out_channels= 128, kernel_size= 4,padding=1) 
        self.max_pooling_5 = nn.MaxPool2d(2,2, return_indices=True)

        #Block 6
        self.conv_6_a = nn.Conv2d(in_channels=128, out_channels= 128, kernel_size= 2,padding=1)
        self.conv_6_b = nn.Conv2d(in_channels=128, out_channels= 128, kernel_size= 3,padding=1) 
        self.conv_6_c = nn.Conv2d(in_channels=128, out_channels= 128, kernel_size= 4,padding=1) 
        self.max_pooling_6 = nn.MaxPool2d(8,8, return_indices=True)

        #Block 7
        self.deconv_6_a = nn.ConvTranspose2d(in_channels=128, out_channels= 128, kernel_size= 1,padding=0)
        self.unpooling_6 = nn.MaxUnpool2d(8, stride=8)

        #Block 8
        self.deconv_7_a = nn.ConvTranspose2d(in_channels=128, out_channels= 128, kernel_size= 5,padding=1)
        self.deconv_7_b = nn.ConvTranspose2d(in_channels=128, out_channels= 128, kernel_size= 1,padding=1)
        self.deconv_7_c = nn.ConvTranspose2d(in_channels=128, out_channels= 128, kernel_size= 3,padding=1)
        self.unpooling_7 = nn.MaxUnpool2d(2, stride=2)

        #Block 9
        self.deconv_8_a = nn.ConvTranspose2d(in_channels=128, out_channels= 64, kernel_size= 8,padding=2)
        self.deconv_8_b = nn.ConvTranspose2d(in_channels=64, out_channels= 64, kernel_size= 3,padding=1)
        self.deconv_8_c = nn.ConvTranspose2d(in_channels=64, out_channels= 64, kernel_size= 6,padding=4)
        self.unpooling_8 = nn.MaxUnpool2d(2, stride=2)

        #Block 10
        self.deconv_9_a = nn.ConvTranspose2d(in_channels=64, out_channels= 32, kernel_size= 2,padding=0)
        self.deconv_9_b = nn.ConvTranspose2d(in_channels=32, out_channels= 32, kernel_size= 11,padding=4)
        self.deconv_9_c = nn.ConvTranspose2d(in_channels=32, out_channels= 32, kernel_size= 16,padding=9)
        self.unpooling_9 = nn.MaxUnpool2d(2, stride=2)

        #Block 11
        self.deconv_10_a = nn.ConvTranspose2d(in_channels=32, out_channels= 16, kernel_size= 3,padding=0)
        self.deconv_10_b = nn.ConvTranspose2d(in_channels=16, out_channels= 16, kernel_size= 2,padding=0)
        self.deconv_10_c = nn.ConvTranspose2d(in_channels=16, out_channels= 16, kernel_size= 4,padding=3)
        self.unpooling_10 = nn.MaxUnpool2d(2, stride=2)

        #Block 12
        self.deconv_11_a = nn.ConvTranspose2d(in_channels=16, out_channels= 4, kernel_size= 5,padding=2)
        self.unpooling_11 = nn.MaxUnpool2d(2, stride=2)
        self.conv_last = nn.Conv2d(in_channels=4, out_channels= self.class_number, kernel_size= 11,padding=5)  
        self.fc_1 = nn.Linear(in_features = image_size[2] , out_features = image_size[2])
        print('number of parameters : convdeconv : ',sum(p.numel() for p in self.parameters() if p.requires_grad))


    def forward(self,x):
        #Block 1
        x = F.relu(self.conv_1_a(x))
        x = F.relu(self.conv_1_b(x))
        x, indices_1 = self.max_pooling_1(x)
        
        #Block 2
        x = F.relu(self.conv_2_a(x))
        x = F.relu(self.conv_2_b(x))
        x, indices_2 = self.max_pooling_2(x)
        
        #Block 3
        x = F.relu(self.conv_3_a(x))
        x = F.relu(self.conv_3_b(x))
        x = F.relu(self.conv_3_c(x))
        x, indices_3 = self.max_pooling_3(x)
        
        #Block 4
        x = F.relu(self.conv_4_a(x))
        x = F.relu(self.conv_4_b(x))
        x = F.relu(self.conv_4_c(x))
        x, indices_4 = self.max_pooling_4(x)
        
        #Block 5
        x = F.relu(self.conv_5_a(x))
        x = F.relu(self.conv_5_b(x))
        x = F.relu(self.conv_5_c(x))
        x, indices_5 = self.max_pooling_5(x)
        
        #Block 6
        x = F.relu(self.conv_6_a(x))
        x = F.relu(self.conv_6_b(x))
        x = F.relu(self.conv_6_c(x))
        x, indices_6 = self.max_pooling_6(x)
        
        #Block 7
        x = F.relu(self.deconv_6_a(x))
        x = self.unpooling_6(x, indices_6)
        
        #Block 8
        x = F.relu(self.deconv_7_a(x))
        x = F.relu(self.deconv_7_b(x))
        x = F.relu(self.deconv_7_c(x))
        x = self.unpooling_7(x, indices_5)
        
        #Block 9
        x = F.relu(self.deconv_8_a(x))
        x = F.relu(self.deconv_8_b(x))
        x = F.relu(self.deconv_8_c(x))
        x = self.unpooling_8(x, indices_4)
        
        #Block 10
        x = F.relu(self.deconv_9_a(x))
        x = F.relu(self.deconv_9_b(x))
        x = F.relu(self.deconv_9_c(x))
        x = self.unpooling_9(x, indices_3)
        
        #Block 11
        x = F.relu(self.deconv_10_a(x))
        x = F.relu(self.deconv_10_b(x))
        x = F.relu(self.deconv_10_c(x))
        
        #Block 12
        x = self.unpooling_10(x, indices_2)
        x = F.relu(self.deconv_11_a(x))
        x = self.unpooling_11(x, indices_1)
        x = F.relu(self.conv_last(x))

        #Block output
        x = F.relu(self.fc_1(x))
        x = softmax_custom(x)
        return  x