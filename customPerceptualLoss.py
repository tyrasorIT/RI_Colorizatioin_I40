import torch.nn as nn
import torchvision.models as models
import torch
from kornia.color import lab_to_rgb
import torch.nn.functional as F

class PerceptualLoss(nn.Module):
    def __init__(self, layers=12):
        super().__init__()
        #vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        try:
            vgg = models.vgg19(
                weights=models.VGG19_Weights.DEFAULT
            ).features
        except AttributeError:
            vgg = models.vgg19(pretrained=True).features
        self.slice = nn.Sequential(*[vgg[i] for i in range(layers)])
        self.slice.eval()
        for p in self.slice.parameters():
            p.requires_grad = False

        self.criterion = nn.L1Loss()

    def forward(self, output, target, L):

        L_lab       = L * 100.0
        pred_ab_lab = output * 128.0
        tgt_ab_lab  = target * 128.0

        predLAB = torch.cat([L_lab, pred_ab_lab], dim=1)
        targetLAB = torch.cat([L_lab, tgt_ab_lab], dim=1)

        predRGB = lab_to_rgb(predLAB)
        targetRGB = lab_to_rgb(targetLAB)

        f_out = self.slice(predRGB)
        f_tar = self.slice(targetRGB)

        return self.criterion(f_out, f_tar)
    
class PerceptualLoss2(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        # use first few layers as feature extractor
        self.features = nn.Sequential(*list(vgg.children())[:16]).eval()
        for p in self.features.parameters():
            p.requires_grad = False

        # VGG expects normalized [0,1] with ImageNet mean/std
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pred_rgb, tgt_rgb):
        """
        pred_rgb, tgt_rgb: (N,3,H,W) in [0,1]
        """
        # normalize for VGG
        x = (pred_rgb - self.mean) / self.std
        y = (tgt_rgb - self.mean) / self.std

        fx = self.features(x)
        fy = self.features(y)

        return F.l1_loss(fx, fy)