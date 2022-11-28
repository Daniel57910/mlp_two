import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class ConvolutionalProcessingBlockDM(nn.Module):

    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation):
        super(ConvolutionalProcessingBlockDM, self).__init__()

        """
        Iitialize the convolutional processing block
            :param input_shape: tuple of ints, shape of input tensor (batch_size, channels, height, width)
            :param num_filters: int, number of num filters
            :param kernel_size: int, size of convolutional kernel
            :param padding: int, padding of convolutional kernel
            :param bias: bool, whether to use bias in convolutional layers
            :param dilation: int, dilation of convolutional kernel
        """
        self.input_filters = input_shape[1]
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.dilation = dilation

        self.layer_dict = nn.ModuleDict({
            'conv1': nn.Conv2d(self.input_filters, self.num_filters, self.kernel_size, padding=self.padding, bias=self.bias, dilation=self.dilation),
            'batch_norm1': nn.BatchNorm2d(self.num_filters),
            'lru1': nn.LeakyReLU(),
            'conv2': nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, padding=self.padding, bias=self.bias, dilation=self.dilation),
            'batch_norm2': nn.BatchNorm2d(self.num_filters),
            'lru2': nn.LeakyReLU(),
        })
    
            
    def forward(self, x):
        keys = list(self.layer_dict.keys())
        for key in keys:
            x = self.layer_dict[key](x)
        return x    


class ResidualConnectionBlockDM(nn.Module):

    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation):
        super(ResidualConnectionBlockDM, self).__init__()


        self.input_filters = input_shape[1]
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.dilation = dilation

        self.layer_dict = nn.ModuleDict({
            'conv1': nn.Conv2d(self.input_filters, self.num_filters, self.kernel_size, padding=self.padding, bias=self.bias, dilation=self.dilation),
            'batch_norm1': nn.BatchNorm2d(self.num_filters),
            'lru1': nn.LeakyReLU(),
            'conv2': nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, padding=self.padding, bias=self.bias, dilation=self.dilation),
            'batch_norm2': nn.BatchNorm2d(self.num_filters),
        })

        self.output_activation = nn.LeakyReLU()
    
    def forward(self, x):
        """
        Basic implementation of residual connection where identity is added to the output of the convolutional block.
        Additional support for skip connections where the number of inputs does not match the number of outputs.
        """
        x_identity = x
        keys = list(self.layer_dict.keys())
        for key in keys:
            x = self.layer_dict[key](x)
        
        if x_identity.shape != x.shape:
            x_identity = nn.Conv2d(
                    self.input_filters, self.num_filters, self.kernel_size, padding=self.padding, bias=self.bias, dilation=self.dilation
            )(x_identity)

        return self.output_activation(x + x_identity)

class ConvolutionalReductionBlockDM(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation, reduction_factor):
        super(ConvolutionalReductionBlockDM, self).__init__()

        """
        Iitialize the convolutional processing block
            :param input_shape: tuple of ints, shape of input tensor (batch_size, channels, height, width)
            :param num_filters: int, number of num filters
            :param kernel_size: int, size of convolutional kernel
            :param padding: int, padding of convolutional kernel
            :param bias: bool, whether to use bias in convolutional layers
            :param dilation: int, dilation of convolutional kernel
            :param reduction_factor: int, factor by which to reduce the height and width of the input tensor
        """
        self.input_filters = input_shape[1]
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.dilation = dilation
    
        self.layer_dict = nn.ModuleDict({
            'conv1': nn.Conv2d(self.input_filters, self.num_filters, self.kernel_size, padding=self.padding, bias=self.bias, dilation=self.dilation),
            'batch_norm1': nn.BatchNorm2d(self.num_filters),
            'lru1': nn.LeakyReLU(),
            'apool': nn.AvgPool2d(reduction_factor),
            'conv2': nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, padding=self.padding, bias=self.bias, dilation=self.dilation),
            'batch_norm2': nn.BatchNorm2d(self.num_filters),
            'lru2': nn.LeakyReLU(),
        })
    
    def forward(self, x):
        keys = list(self.layer_dict.keys())
        for key in keys:
            x = self.layer_dict[key](x)
        return x    