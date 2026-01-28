import torch
import torch.nn as nn
import piq
from utils.image import lab_to_rgb_torch
import math

class CombinedColorizationLoss(nn.Module):
    def __init__(self, pixelLossFn, percepLossFn, chromaLossFn, pixel_weight=0.3, percep_weight=0.05, chroma_weight=0.1):
        super().__init__()
        self.pixelLossFn = pixelLossFn
        self.percepLossFn = percepLossFn
        self.chromaLossFn = chromaLossFn
        self.chroma_weight = chroma_weight
        self.pixel_weight = pixel_weight
        self.percep_weight = percep_weight

    def forward(self, predAB, targetAB, L):
        pixel_loss = self.pixelLossFn(predAB, targetAB)
        percep_loss = self.percepLossFn(predAB, targetAB, L)
        chroma_loss = self.chromaLossFn(predAB, targetAB)

        # print(f"Separate losses: pixel: {pixel_loss.item():.4f}, percep: {percep_loss.item():.4f}, chroma: {chroma_loss.item():.4f}")

        total_loss = (self.pixel_weight * pixel_loss + 
                      self.percep_weight * percep_loss + 
                      self.chroma_weight * chroma_loss)
        return total_loss
    
class CombinedColorizationLossCIEDE(nn.Module):
    """
    Pixel-wise color loss in LAB (CIEDE2000) + perceptual loss on RGB.
    Pixel term is chroma-only via shared L.
    """
    def __init__(self, percep_loss: nn.Module, percep_weight: float = 0.3):
        super().__init__()
        self.color_loss = CIEDE2000ColorLoss()
        self.percep_loss = percep_loss
        self.percep_weight = percep_weight

    def forward(self, pred_ab, target_ab, L):
        """
        pred_ab:   (N, 2, H, W)
        target_ab: (N, 2, H, W)
        L:         (N, 1, H, W)
        """
        # 1) Color loss in LAB space (chroma-only)
        pixel_loss = self.color_loss(pred_ab, target_ab, L)

        # 2) Perceptual loss on RGB
        with torch.no_grad():
            real_rgb = lab_to_rgb_torch(L, target_ab)  # (N, 3, H, W) in [0,1]
        pred_rgb = lab_to_rgb_torch(L, pred_ab)

        percep_loss = self.percep_loss(pred_rgb, real_rgb)

        total = pixel_loss + self.percep_weight * percep_loss
        return total

# ---- CIEDE2000 core ---------------------------------------------------------

def _deg2rad(x: float) -> float:
    return x * math.pi / 180.0


def ciede2000_torch(Lab1: torch.Tensor, Lab2: torch.Tensor) -> torch.Tensor:
    """
    Lab1, Lab2: (N, 3, H, W)
    Returns mean ΔE2000 over N, H, W.
    """
    L1, a1, b1 = Lab1[:, 0], Lab1[:, 1], Lab1[:, 2]
    L2, a2, b2 = Lab2[:, 0], Lab2[:, 1], Lab2[:, 2]

    kL = kC = kH = 1.0

    C1 = torch.sqrt(a1**2 + b1**2)
    C2 = torch.sqrt(a2**2 + b2**2)
    Cbar = 0.5 * (C1 + C2)

    G = 0.5 * (1 - torch.sqrt((Cbar**7) / (Cbar**7 + 25**7)))

    a1p = (1 + G) * a1
    a2p = (1 + G) * a2

    C1p = torch.sqrt(a1p**2 + b1**2)
    C2p = torch.sqrt(a2p**2 + b2**2)

    h1p = torch.atan2(b1, a1p)
    h2p = torch.atan2(b2, a2p)

    two_pi = 2 * math.pi
    h1p = torch.where(h1p < 0, h1p + two_pi, h1p)
    h2p = torch.where(h2p < 0, h2p + two_pi, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = torch.where(dhp >  math.pi, dhp - two_pi, dhp)
    dhp = torch.where(dhp < -math.pi, dhp + two_pi, dhp)

    dHp = 2 * torch.sqrt(C1p * C2p) * torch.sin(dhp / 2)

    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)

    hsum = h1p + h2p
    habsp = torch.abs(h1p - h2p)

    hbarp = torch.where(
        (C1p * C2p) == 0,
        hsum,
        torch.where(
            habsp <= math.pi,
            0.5 * hsum,
            torch.where(
                hsum < two_pi,
                0.5 * (hsum + two_pi),
                0.5 * (hsum - two_pi),
            ),
        ),
    )

    T = (1
         - 0.17 * torch.cos(hbarp - _deg2rad(30))
         + 0.24 * torch.cos(2 * hbarp)
         + 0.32 * torch.cos(3 * hbarp + _deg2rad(6))
         - 0.20 * torch.cos(4 * hbarp - _deg2rad(63)))

    SL = 1 + (0.015 * (Lbarp - 50) ** 2) / torch.sqrt(20 + (Lbarp - 50) ** 2)
    SC = 1 + 0.045 * Cbarp
    SH = 1 + 0.015 * Cbarp * T

    Rt = (-2 * torch.sqrt((Cbarp**7) / (Cbarp**7 + 25**7))
          * torch.sin(_deg2rad(60) * torch.exp(-((hbarp - _deg2rad(275)) / _deg2rad(25)) ** 2)))

    dE = torch.sqrt(
        (dLp / (kL * SL))**2 +
        (dCp / (kC * SC))**2 +
        (dHp / (kH * SH))**2 +
        Rt * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )

    return dE.mean()


class CIEDE2000ColorLoss(nn.Module):
    """
    LAB color loss using CIEDE2000.

    pred_ab, target_ab are in [-1, 1]
    L is in [0, 1]

    L is reused for both pred and target LAB, so the loss
    is effectively *chroma-only* (no penalty on L).
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_ab: torch.Tensor,
                target_ab: torch.Tensor,
                L: torch.Tensor) -> torch.Tensor:
        # Undo normalization to real LAB ranges
        L_lab       = L * 100.0         # [0, 100]
        pred_ab_lab = pred_ab * 128.0   # [-128, 128]
        tgt_ab_lab  = target_ab * 128.0

        pred_lab = torch.cat([L_lab, pred_ab_lab], dim=1)  # (N,3,H,W)
        tgt_lab  = torch.cat([L_lab, tgt_ab_lab], dim=1)

        return ciede2000_torch(pred_lab, tgt_lab)