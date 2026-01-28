from torch import nn
import torch
from torchvision import models

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 2, 4, stride=2, padding=1),  # no tanh
            nn.Tanh()
        )

    def forward(self, x):
        code = self.encoder(x)
        return self.decoder(code)
    
class UNetAutoencoder(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, out_ch, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d3 = self.dec3(e3)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return d1

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
        return out

class ResNetUNetColorization(nn.Module):
    """
    Colorization network with a ResNet-18 encoder (pretrained on ImageNet)
    and a U-Net style decoder. Input: L (N,1,H,W). Output: ab (N,2,H,W).
    """
    def __init__(self, out_ch=2, pretrained=True):
        super().__init__()

        if pretrained:
            try:
                # New torchvision (>= 0.13)
                resnet = models.resnet18(
                    weights=models.ResNet18_Weights.IMAGENET1K_V1
                )
            except AttributeError:
                # Old torchvision (< 0.13)
                resnet = models.resnet18(pretrained=True)
        else:
            resnet = models.resnet18(pretrained=False)

        # Encoder: take layers from ResNet
        self.initial = nn.Sequential(
            resnet.conv1,  # 64, H/2
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool           # 64, H/4
        self.layer1 = resnet.layer1             # 64, H/4
        self.layer2 = resnet.layer2             # 128, H/8
        self.layer3 = resnet.layer3             # 256, H/16
        self.layer4 = resnet.layer4             # 512, H/32

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # H/16
        self.dec4 = self._dec_block(256 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # H/8
        self.dec3 = self._dec_block(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # H/4
        self.dec2 = self._dec_block(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)    # H/2
        self.dec1 = self._dec_block(64 + 64, 64)

        # Final upsample to full resolution (H)
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)    # H
        self.dec0 = self._dec_block(32, 32)

        self.out_conv = nn.Conv2d(32, out_ch, kernel_size=1)

    @staticmethod
    def _dec_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, L):
        """
        L: (N,1,H,W) normalized [0,1]
        """
        # Repeat L to 3 channels for ResNet
        x = L.repeat(1, 3, 1, 1)  # (N,3,H,W)

        x0 = self.initial(x)      # (N,64,H/2,W/2)
        x1 = self.maxpool(x0)     # (N,64,H/4,W/4)
        x1 = self.layer1(x1)      # (N,64,H/4,W/4)
        x2 = self.layer2(x1)      # (N,128,H/8,W/8)
        x3 = self.layer3(x2)      # (N,256,H/16,W/16)
        x4 = self.layer4(x3)      # (N,512,H/32,W/32)

        # Decoder with skips
        d4 = self.up4(x4)                         # (N,256,H/16,W/16)
        d4 = self.dec4(torch.cat([d4, x3], 1))    # concat with x3

        d3 = self.up3(d4)                         # (N,128,H/8,W/8)
        d3 = self.dec3(torch.cat([d3, x2], 1))

        d2 = self.up2(d3)                         # (N,64,H/4,W/4)
        d2 = self.dec2(torch.cat([d2, x1], 1))

        d1 = self.up1(d2)                         # (N,64,H/2,W/2)
        d1 = self.dec1(torch.cat([d1, x0], 1))

        d0 = self.up0(d1)                         # (N,32,H,W)
        d0 = self.dec0(d0)

        out = self.out_conv(d0)                   # (N,2,H,W)
        return out
