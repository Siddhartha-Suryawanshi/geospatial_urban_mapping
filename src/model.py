import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):
        super(ASPPModule, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        for dilation in dilations:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3Plus(nn.Module):
    def __init__(self, n_classes, n_channels=3):
        super(DeepLabV3Plus, self).__init__()
        resnet = models.resnet50(weights='DEFAULT')
        
        if n_channels != 3:
            resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        self.encoder_low = nn.Sequential(*list(resnet.children())[:5])
        self.encoder_high = nn.Sequential(*list(resnet.children())[5:8])
        
        self.aspp = ASPPModule(2048, 256, [6, 12, 18])
        
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1))

    def forward(self, x):
        input_size = x.shape[2:]
        low_level_feat = self.encoder_low(x)
        high_level_feat = self.encoder_high(low_level_feat)
        
        x = self.aspp(high_level_feat)
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        
        low_level_feat = self.low_level_conv(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.final_conv(x)
        
        return F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)