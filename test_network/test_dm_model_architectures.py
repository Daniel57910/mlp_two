import pytest
import torch
from pytorch_mlp_framework.model_architectures_dm import (
    ConvolutionalProcessingBlockDM
)

import numpy as np

def test_import_convolutional_processing_block():
    assert ConvolutionalProcessingBlockDM is not None


def test_basic_convolution_block_batch_norm():
    output_filters = 6
    stride=3
    padding = 1
    bias = True
    dilation = 1

    """
    Create random input tensor
    Representing a batch of 4 convolutions of 32*32 pixels with 3 channels
    """
    sample = torch.randn(4, 3, 32, 32)

    block = ConvolutionalProcessingBlockDM(
        sample.shape,
        output_filters,
        stride,
        padding,
        bias,
        dilation
    )


    out = block(sample)
    assert out.shape==(4, 6, 32, 32)
    out = out.detach().numpy()

    """
    assert means and variances all channels similar
    """
    channel_means = np.mean(out, axis=(0, 2, 3))
    channel_vars = np.var(out, axis=(0, 2, 3))

    assert np.allclose(channel_means, np.mean(channel_means), atol=1e-2)
    assert np.allclose(channel_vars, np.mean(channel_vars), atol=1e-1)