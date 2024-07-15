import torch
import torch.nn as nn


class Resnet50_Adaptor(nn.Module) :

    def __init__(self, input_channels, output_channels) :
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1))

    def forward(self, x) :
        return self.conv(x)