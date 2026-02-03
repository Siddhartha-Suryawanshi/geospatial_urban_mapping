import torch
import torch.nn as nn
from .attention import AttentionGate  # Essential: imports from attention.py

class ConvBlock(nn.Module):
    """Standard double convolution block for U-Net."""
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, n_channels=2, n_classes=4):
        super(AttentionUNet, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = ConvBlock(n_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        # Decoder (Upsampling with Attention)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = ConvBlock(128, 64)
        
        # Final Classification
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Down Path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        
        # Up Path with Attention
        d1 = self.up2(e2)
        # Attention Gate filters the skip connection e1
        a1 = self.att1(g=d1, x=e1)
        
        # Concatenate and Classify
        out = self.dec1(torch.cat((a1, d1), dim=1))
        return self.final(out)