import torch
from skimage import color
import numpy as np

def rgb_to_lab(img):
    img = img.permute(1, 2, 0).numpy()
    lab = color.rgb2lab(img).astype("float32")
    return lab

def split_and_normalize(lab):
    l = lab[:, :, 0]
    ab = lab[:, :, 1:]

    l = (l / 50.0) - 1.0    # Changed from l / 100.0
    ab = ab / 110.0          # This was already correct

    l = torch.tensor(l).unsqueeze(0)
    ab = torch.tensor(ab).permute(2, 0, 1)
    
    return l, ab

def lab_to_rgb_torch(L, ab):
    """
    L:  (N,1,H,W) in [0,1]
    ab: (N,2,H,W) in [-1,1]
    returns: (N,3,H,W) in [0,1]
    """

    # Undo normalization
    L = L * 100.0
    a = ab[:, 0:1] * 110.0
    b = ab[:, 1:2] * 110.0

    a = torch.clamp(a, -128, 127)
    b = torch.clamp(b, -128, 127)
    L = torch.clamp(L, 0, 100)

    # --- Convert LAB to XYZ (D65) ---
    y = (L + 16.0) / 116.0
    x = a / 500.0 + y
    z = y - b / 200.0

    x3 = x**3
    y3 = y**3
    z3 = z**3

    epsilon = 0.008856
    kappa = 903.3

    x = torch.where(x3 > epsilon, x3, (x - 16/116) / 7.787)
    y = torch.where(y3 > epsilon, y3, (y - 16/116) / 7.787)
    z = torch.where(z3 > epsilon, z3, (z - 16/116) / 7.787)

    # Reference white D65
    X = x * 95.047
    Y = y * 100.000
    Z = z * 108.883

    # --- Convert XYZ to RGB ---
    X /= 100
    Y /= 100
    Z /= 100

    r =  3.2406*X - 1.5372*Y - 0.4986*Z
    g = -0.9689*X + 1.8758*Y + 0.0415*Z
    b =  0.0557*X - 0.2040*Y + 1.0570*Z

    # Gamma correction
    def gamma(inv):
        return torch.where(inv > 0.0031308,
                           1.055 * torch.pow(inv, 1/2.4) - 0.055,
                           12.92 * inv)

    r = gamma(r)
    g = gamma(g)
    b = gamma(b)

    # Stack into an image
    rgb = torch.cat([r, g, b], dim=1)
    return rgb.clamp(0.0, 1.0)

def lab_to_rgb(L, ab):
    # L: (1,H,W), [0,1]
    # ab: (2,H,W), [-1,1]
    
    lab = np.zeros((L.shape[1], L.shape[2], 3), dtype=np.float32)
    lab[:, :, 0] = L[0] * 100.0
    lab[:, :, 1:] = ab.transpose(1, 2, 0) * 110.0
    
    rgb = color.lab2rgb(lab)  # returns [0,1] float
    return rgb