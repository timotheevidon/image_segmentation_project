# general imports
import torch.nn as nn
import torch.nn.functional as F

# imports from others files
from utils import softmax_custom

class VanillaCNN(nn.Module):
    def __init__(self, class_number, image_size):
        super(VanillaCNN,self).__init__()
        self.name = 'vanilla'
        self.image_size = image_size
        self.class_number = class_number

        #Block 1
        self.conv_1 = nn.Conv2d(in_channels= self.image_size[0], out_channels= 3, kernel_size= 5,padding=2)
        self.conv_2 = nn.Conv2d(in_channels= 3, out_channels= 3, kernel_size= 7,padding=4)
        self.conv_3 = nn.Conv2d(in_channels= 3, out_channels= self.class_number, kernel_size= 11,padding=4)

        #Block output
        self.fc_1 = nn.Linear(in_features = image_size[2] , out_features = image_size[2])
        print('number of parameters : vanilla : ',sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self,x):
        # Block 1
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))

        #Block output
        x = F.relu(self.fc_1(x))
        x = softmax_custom(x)
        return  x