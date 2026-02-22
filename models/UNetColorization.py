from torch import nn
import torch

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNetColorization(nn.Module):
    """
    U-Net that takes L (1 channel) and predicts ab (2 channels).
    """
    def __init__(self, in_ch=1, out_ch=2, base_ch=64):
        super().__init__()

        # Encoder
        self.enc1 = conv_block(in_ch, base_ch)          # 1 -> 64
        self.pool1 = nn.MaxPool2d(2)                    # H/2

        self.enc2 = conv_block(base_ch, base_ch * 2)    # 64 -> 128
        self.pool2 = nn.MaxPool2d(2)                    # H/4

        self.enc3 = conv_block(base_ch * 2, base_ch * 4) # 128 -> 256
        self.pool3 = nn.MaxPool2d(2)                     # H/8

        # Bottleneck
        self.bottleneck = conv_block(base_ch * 4, base_ch * 8)  # 256 -> 512

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)  # concat 256 + 256

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)  # concat 128 + 128

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)      # concat 64 + 64

        # Output: 2 channels (ab), NO Tanh here
        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)
        self.tanh = nn.Tanh()                             # constrain output to [-1, 1], correct range for LAB space

    def forward(self, x):
        # x: (N, 1, H, W) L-channel
        e1 = self.enc1(x)         # (N, 64, H, W)
        p1 = self.pool1(e1)       # (N, 64, H/2, W/2)

        e2 = self.enc2(p1)        # (N, 128, H/2, W/2)
        p2 = self.pool2(e2)       # (N, 128, H/4, W/4)

        e3 = self.enc3(p2)        # (N, 256, H/4, W/4)
        p3 = self.pool3(e3)       # (N, 256, H/8, W/8)

        b = self.bottleneck(p3)   # (N, 512, H/8, W/8)

        u3 = self.up3(b)          # (N, 256, H/4, W/4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)         # (N, 128, H/2, W/2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)         # (N, 64, H, W)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.out_conv(d1)   # (N, 2, H, W)
        out = self.tanh(out)      # constrain output to [-1, 1], correct range for LAB space

        return out