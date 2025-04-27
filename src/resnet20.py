import torch
import torch.nn as nn
from basic_block import BasicBlock

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.block1_1 = BasicBlock(16, 16)
        self.block1_2 = BasicBlock(16, 16)
        self.block1_3 = BasicBlock(16, 16)

        self.block2_1 = BasicBlock(
            in_channels=16,
            out_channels=32,
            stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(32)
            )
        )
        self.block2_2 = BasicBlock(32, 32)
        self.block2_3 = BasicBlock(32, 32)

        self.block3_1 = BasicBlock(
            in_channels=32,
            out_channels=64,
            stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64)
            )
        )
        self.block3_2 = BasicBlock(64, 64)
        self.block3_3 = BasicBlock(64, 64)

        self.avgpooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block1_3(x)

        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)

        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)

        x = self.avgpooling(x)
        x = torch.flatten(x ,1)
        x = self.fc(x)

        return x