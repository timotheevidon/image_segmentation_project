# General Imports
import torch.nn as nn
import torch.nn.functional as F

# imports from others files
from utils import softmax_custom
    
class PoolUnpool(nn.Module):
    def __init__(self, class_number, image_size):
        super(PoolUnpool,self).__init__()
        self.name = 'poolUnpool'
        self.image_size = image_size
        self.class_number = class_number

        #Block 1
        self.conv_1 = nn.Conv2d(in_channels= self.image_size[0], out_channels= 3, kernel_size= 5,padding=2)
        self.conv_2 = nn.Conv2d(in_channels= 3, out_channels= 8, kernel_size= 7,padding=4)
        self.conv_3 = nn.Conv2d(in_channels= 8, out_channels= self.class_number, kernel_size= 11,padding=4)
        self.maxpool_1 = nn.MaxPool2d(2, stride=2, return_indices=True)

        #Block 2
        self.conv_6_a = nn.Conv2d(in_channels=self.class_number, out_channels= self.class_number, kernel_size= 2,padding=1)
        self.conv_6_b = nn.Conv2d(in_channels=self.class_number, out_channels= self.class_number, kernel_size= 3,padding=1) 
        self.conv_6_c = nn.Conv2d(in_channels=self.class_number, out_channels= self.class_number, kernel_size= 4,padding=1) 
        self.maxpool_2 = nn.MaxPool2d(4,2, return_indices=True)

        #Block 3
        self.deconv_6_a = nn.ConvTranspose2d(in_channels=self.class_number, out_channels= self.class_number, kernel_size= 3,padding=1)
        self.maxunpool_1 = nn.MaxUnpool2d(4, stride=2)
        self.maxunpool_2 = nn.MaxUnpool2d(2, stride=2)
        self.fc_1 = nn.Linear(in_features = image_size[2] , out_features = image_size[2])

        print('number of parameters : poolUnpool : ',sum(p.numel() for p in self.parameters() if p.requires_grad))
        
    def forward(self,x):

        #Block 1
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x, indices_1 = self.maxpool_1(x)

        #Block 2
        x = F.relu(self.conv_6_a(x))
        x = F.relu(self.conv_6_b(x))
        x = F.relu(self.conv_6_c(x))
        x, indices_6 = self.maxpool_2(x)

        #Block 3
        x = self.maxunpool_1(x, indices_6)
        x = self.maxunpool_2(x, indices_1)
        x = F.relu(self.fc_1(x))
        x = softmax_custom(x)
        return  x