"""Architecture module - contains all neural network architectures."""

from .base import BaseArchitecture
from .simple_cnn import SimpleCNN
from .resnet import ResNet
from .fc_net import FullyConnectedNet

__all__ = [
    'BaseArchitecture',
    'SimpleCNN',
    'ResNet',
    'FullyConnectedNet',
]
