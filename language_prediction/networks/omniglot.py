import numpy as np
import torch
from torch import nn
import torch.nn.functional as F  # pylint: disable=unused-import

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4

class OmniglotCNN(nn.Module):
    """Container for ProtoNet weights and image-to-latent computation."""

    def __init__(self, num_outputs=5):
        """Inits ProtoNetNetwork.

        The network consists of four convolutional blocks, each comprising a
        convolution layer, a batch normalization layer, ReLU activation, and 2x2
        max pooling for downsampling. There is an additional flattening
        operation at the end.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.
        """
        super().__init__()
        layers = []
        in_channels = NUM_INPUT_CHANNELS
        for _ in range(NUM_CONV_LAYERS):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    NUM_HIDDEN_CHANNELS,
                    (KERNEL_SIZE, KERNEL_SIZE),
                    padding='same'
                )
            )
            layers.append(nn.BatchNorm2d(NUM_HIDDEN_CHANNELS))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = NUM_HIDDEN_CHANNELS
        layers.append(nn.Flatten())
        layers.append(nn.Linear(NUM_HIDDEN_CHANNELS, num_outputs))
        self._layers = nn.Sequential(*layers)

    @property
    def action_pad_idx(self):
        return -100
    
    @property
    def lang_pad_idx(self):
        return -100

    def forward(self, obs):
        """Computes the latent representation of a batch of images.

        Args:
            images (Tensor): batch of Omniglot images
                shape (num_images, channels, height, width)

        Returns:
            a Tensor containing a batch of latent representations
                shape (num_images, latents)
        """
        return self._layers(obs['image']), None