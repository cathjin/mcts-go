import torch
from torch import nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block = ConvBlock()

        self.res_blocks = nn.Sequential(
            *[ResBlock() for _ in range(9)]
        )
        self.policy = PolicyHead()
        self.value = ValueHead()

    def forward(self, x):
        out = self.conv_block(x)
        out = self.res_blocks(out)
        p = self.policy(out)
        v = self.value(out)
        return p, v

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1) # change in_channels to take in history and player later
        self.batch_norm = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace = False)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        return out

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=False)

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = out + x
        
        out = self.conv2(x)
        out = self.batch_norm2(out)
        out = self.relu(out)
        return out

class PolicyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(num_features=2)
        self.relu = nn.ReLU(inplace=False)
        self.linear = nn.Linear(in_features=162, out_features=81)
    
    def forward(self, x):
        p = self.conv(x)
        p = self.batch_norm(p)
        p = self.relu(p)
        p = p.view(p.size(0), -1)
        p = self.linear(p)
        return p

class ValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(num_features=1)
        self.linear1 = nn.Linear(in_features=81, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=1)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        v = self.conv(x)
        v = self.batch_norm(v)
        v = self.relu(v)
        v = v.view(v.size(0), -1)
        v = self.linear1(v)
        v = self.relu(v)
        v = self.linear2(v)
        v = torch.tanh(v)
        return v
        