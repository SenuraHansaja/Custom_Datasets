import torch
from torch import nn

class TinyVGG(nn.Module):
    """replicating the tinyVGG architecture"""
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int)->None:
        super.__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channles = input_shape, 
                      out_channels =  hidden_units, 
                      kernel_size = 3, 
                      stride = 1 ,
                      padding = 0),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, 
                      out_channels = hidden_units,
                      kernel_size = 3,
                      stirde = 1,
                      padding =0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride = 2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channles = hidden_units, 
                      out_channels =  hidden_units, 
                      kernel_size = 3, 
                      stride = 1 ,
                      padding = 0),
            nn.ReLU(),
            nn.Conv2d(in_channles = hidden_units, 
                      out_channels =  hidden_units, 
                      kernel_size = 3, 
                      stride = 1 ,
                      padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = hidden_units*13*13,
                      output_features = output_shape)
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)

        return x