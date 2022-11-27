import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalProcessingBlock(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation):
        super(ConvolutionalProcessingBlock, self).__init__()

        self.input_shape = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.dilation = dilation

    