### Margaret Maynard-Reid 6/5/2019

### TODO: define the convolutional neural network architecture
### Margaret: defined a neural network architecture based on this paper:
### Facial Key Points Detection using Deep Convolutional Neural Network - NaimishNet
### https://arxiv.org/pdf/1710.00977.pdf
    
import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # (all) Pooling layer
        self.pool = nn.MaxPool2d(2,2)
        
        ### Convolutional layers ###
        # Layer 2, 1st conv layer
        self.conv1 = nn.Conv2d(1, 32, 4)
        # Layer 6, 2nd conv layer
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Layer 10, 3rd conv layer
        self.conv3 = nn.Conv2d(64, 128, 2)
        # Layer 14 4th conv layer
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        # Batch norm layers
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(256)
        
        # Dropout layers ###
        # Layer 5 1st dropout layer
        self.drop1 = nn.Dropout(p=0.1)        
        # Layer 9 2nd dropout layer
        self.drop2 = nn.Dropout(p=0.2)   
        # Layer 13 3rd dropout layer
        self.drop3 = nn.Dropout(p=0.3)
        # Layer 17 4th dropout layer
        self.drop4 = nn.Dropout(p=0.4)
        # Layer 21 5th dropout layer
        self.drop5 = nn.Dropout(p=0.5)
        # Laye 24 6th dropout layer
        self.drop6 = nn.Dropout(p=0.6)
        
        ### Dense layers ###
        #19 1st Dense layer
        self.dense1 = nn.Linear(256*13*13, 2000)         
        #22 2nd dense layer
        self.dense2 = nn.Linear(2000, 1000)
        #25 3rd dense layer
        self.dense3 = nn.Linear(1000, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # a modified x, having gone through all the layers of your model, should be returned
        
        #1-4
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
                         
        #5
#         x = self.drop1(x)
                      
        #6-8
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
               
        #9
#         x = self.drop2(x)
                      
        #10-12
        x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))
               
        #13
#         x = self.drop3(x)
                      
        #14-16
        x = self.pool(F.relu(self.batchnorm4(self.conv4(x))))
               
        #17
#         x = self.drop4(x)
        
        #18 Flatten              
        x = x.view(x.size(0), -1)
                      
        #19 dense 1 & #20 then activation
        x = F.elu(self.dense1(x))
        
        #21 dropout
#         x = self.drop5(x)
      
        #22 dense 2 & #23 then activation
        x = F.elu(self.dense2(x))
                      
        #24 dropout 6
#         x = self.drop6(x)
                
        #25 dense 3
        x = self.dense3(x)
                      
        return x
