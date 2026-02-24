import torch
from torch import nn
from torchvision import models

class ResNetUNetColorization(nn.Module):
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

        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._dec_block(256 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._dec_block(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._dec_block(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._dec_block(64 + 64, 64)

        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = self._dec_block(32, 32)

        self.out_conv = nn.Conv2d(32, out_ch, kernel_size=1)
        self.tanh = nn.Tanh()

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
        x = L.repeat(1, 3, 1, 1)

        x0 = self.initial(x)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

       
        d4 = self.up4(x4)
        d4 = self.dec4(torch.cat([d4, x3], 1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, x2], 1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x1], 1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x0], 1))

        d0 = self.up0(d1)
        d0 = self.dec0(d0)

        out = self.out_conv(d0)
        out = self.tanh(out)
        return out