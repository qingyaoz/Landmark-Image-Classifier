"""
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Challenge(nn.Module):
    def __init__(self):
        """Define the architecture."""
        super().__init__()

        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # output:  16×16×16
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=2) # output:  64×8×8
        # # Max Pooling Layer Output: 64×4×4
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=5, stride=2, padding=2) # Output: 8×2×2
        # self.fc_1 = nn.Linear(32, 64)
        # self.fc_2 = nn.Linear(64, 2)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # output:  16×32×32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding="same") # output:  64×32×32
        # Max Pooling Layer Output: 64×16×16
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="same") # Output: 128×16×16
        # Max Pooling Layer Output: 128×8×8
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding="same") # Output: 64×8×8
        # Max Pooling Layer Output: 64×4×4
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding="same") # Output: 8×4×4
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(p=0.1)

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        for fc in [self.fc_1, self.fc_2]:
            fc_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / sqrt(fc_in))
            nn.init.constant_(fc.bias, 0.0)

    def forward(self, x):
        N, C, H, W = x.shape

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)    # flatten the tensor to (batch_size, num of feature)
        z = self.fc_1(x)
        z = F.relu(z)
        z = self.fc_2(z)

        return z
