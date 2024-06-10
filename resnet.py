import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, groups=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.BatchNorm2d(inchannel, affine=False),
            nn.MaxPool2d(kernel_size=stride, stride=stride),
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, bias=False, groups=groups),
            nn.SiLU(),
        )
        self.stride = stride
        self.short = nn.Conv2d(inchannel, outchannel, 1)

    def forward(self, x):
        out = self.left(x)
        x = F.interpolate(x, size=(x.size(-2) // self.stride, x.size(-1) // self.stride))
        out = out + self.short(x)

        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 4
        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=2)
        self.layer1 = self.make_layer(ResidualBlock, 4, 1, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 8, 1, stride=1)
        self.layer3 = self.make_layer(ResidualBlock, 8, 1, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 16, 1, stride=1)
        self.layer5 = self.make_layer(ResidualBlock, 16, 1, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(16, 6)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(6, num_classes)
        
        self.a = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(1.), requires_grad=True)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for stride in strides:
            if self.inchannel == 1:
                layers.append(ResidualBlock(self.inchannel, channels, stride, 1))
            else:
                layers.append(ResidualBlock(self.inchannel, channels, stride, self.inchannel // 2))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(x)
        out1 = self.layer1(out)
        out1 = self.layer2(out1)
        out2 = self.layer3(out1) + self.a * F.interpolate(out1, size=(out1.size(-2) // 2, out1.size(-1) // 2))
        out2 = self.layer4(out2)
        out3 = self.layer5(out2) + self.b * F.interpolate(out2, size=(out2.size(-2) // 2, out2.size(-1) // 2))
        out = self.avg_pool(out3)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.tanh(out)
        out = self.fc2(out)
        return out