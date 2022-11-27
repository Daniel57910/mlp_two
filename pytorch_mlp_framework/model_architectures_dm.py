import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalProcessingBlockDM(nn.Module):
    def __init__(self, input_shape, output_filters, kernel_size, padding, bias, dilation):
        super(ConvolutionalProcessingBlockDM, self).__init__()

        """
        Iitialize the convolutional processing block
            :param input_shape: tuple of ints, shape of input tensor (batch_size, channels, height, width)
            :param output_filters: int, number of output filters
            :param kernel_size: int, size of convolutional kernel
            :param padding: int, padding of convolutional kernel
            :param bias: bool, whether to use bias in convolutional layers
            :param dilation: int, dilation of convolutional kernel
        """
        self.input_filters = input_shape[1]
        self.output_filters = output_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.dilation = dilation
    
        self.block_one = nn.Sequential(
            nn.Conv2d(
                self.input_filters, 
                self.output_filters, 
                self.kernel_size, 
                padding=self.padding, 
                bias=self.bias,
                dilation=self.dilation),
            nn.BatchNorm2d(self.output_filters),
            nn.LeakyReLU(),
        )

        self.block_two = nn.Sequential(
            nn.Conv2d(
                self.output_filters,
                self.output_filters,
                self.kernel_size,
                padding=self.padding,
                bias=self.bias,
                dilation=self.dilation
            ), 
            nn.BatchNorm2d(self.output_filters),
            nn.LeakyReLU(),
        )  
            
    def forward(self, x):
        x = self.block_one(x)
        x = self.block_two(x)
        return x