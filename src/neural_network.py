import torch
from torch import nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=False)

        self.res_blocks = nn.Sequential(
            *[ResBlock() for _ in range(20)]
        )
        
        self.p_conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.p_batch_norm = nn.BatchNorm2d(num_features=3)
        self.p_out = nn.Linear(in_features = 243, out_features=82)

        self.v_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding = 1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_out = nn.Linear(in_features=81, out_features=1)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)    
        out = self.res_blocks(out)
        p = self.p_conv(out)
        p = self.p_batch_norm(p)
        p = self.relu(p)
        p = p.view(p.size(0), -1)
        p = self.p_out(p)
        p = F.softmax(p, dim = 1)

        v = self.v_conv(out)
        v = self.v_bn(v)
        v = v.view(v.size(0), -1)
        v = self.v_out(v)
        v = torch.tanh(v)

        return p, v

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)

        out = out + x
        out = self.relu(out)
        return out
