import torch
import torch.nn as nn

class Model(nn.Module):

  def __init__(self, num_classes):
    super(Model, self).__init__()

    # Image shape : (batch_size, kernel, height, width) > (10, 3, 64, 64)  
    '''
    Shape of image in next layer = ((w-f)+2p)/s + 1 = ((150-3)+2*(1))/1 + 1 = 64
      where,  w : width
              f : kernel_size
              p : padding
              s : stride
    '''
    
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
    # shape = (10, 12, 64, 64)
    self.bn1 = nn.BatchNorm2d(num_features=12) # Does not affects the shape   
    self.relu1 = nn.ReLU()                     # Does not affects the shape   
    self.pool1 = nn.MaxPool2d(2, 2)             # Reduces the image size by a factor of 2
    # Shape = (10, 12, 32, 32)

    self.conv2 = nn.Conv2d(12, 20, 3, 1, 1)
    # Shape = (10, 20, 32, 32)
    self.relu2 = nn.ReLU()
    # self.pool2 = nn.MaxPool2d(2, 2)
    # Shape = (10, 20, 32, 32)

    self.conv3 = nn.Conv2d(20, 12, 3, 1, 1)
    # shape = (10, 12, 32, 32)
    self.bn3 = nn.BatchNorm2d(num_features=12) # Does not affects the shape   
    self.relu3 = nn.ReLU()                     # Does not affects the shape   

    self.fc1 = nn.Linear(12*64*64, 64)
    self.fc2 = nn.Linear(64, 16)
    self.fc3 = nn.Linear(16, 2)

  def forward(self, x):

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.pool1(x)

    x = self.conv2(x)
    x = self.relu2(x)
    # x = self.pool2(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu3(x)

    # Above output is in matrix form, with shape (10, 12, 32, 32)
    x = x.view(-1, 12*64*64)

    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)

    return x