import pytest
from pytorch_mlp_framework.model_architectures_dm import (
    ConvolutionalProcessingBlock
)


def test_import_convolutional_processing_block():
    assert ConvolutionalProcessingBlock is not None