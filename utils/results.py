import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from skimage import color
import os
import time
import PIL
from torchvision import transforms
from external.colorization.colorizers import *
from config import CONFIG
import warnings

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    L = torch.clamp(L, -1., 1.)
    ab = torch.clamp(ab, -1., 1.)

    L = L.detach().cpu()
    ab = ab.detach().cpu()

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1)
    Lab = Lab.permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    with warnings.catch_warnings(): #Annoying out of bound warnings #Under 1%
        warnings.simplefilter("ignore", UserWarning)
        for img in Lab:
            img_rgb = color.lab2rgb(img)
            rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def show_results3(device, model: torch.nn.Module, data, idx, size, output_dir=CONFIG.RESULTS_DIR):
    model.generatorNet.eval()
    l, ab = data[idx]

    os.makedirs(output_dir, exist_ok=True)

    Lb = l.unsqueeze(0).to(device)  

    with torch.no_grad():
        # Predict ab

        pred_ab = model.generatorNet(Lb)

    # Predicted RGB
    pred_rgb = lab_to_rgb(Lb.cpu(), pred_ab.cpu())[0]  # [H,W,3] (assuming your helper does this)

    # Real RGB (optional comparison)
    real_rgb = lab_to_rgb(Lb.cpu(), ab.unsqueeze(0).cpu())[0]

    # L for display (optional): convert [-1,1] -> [0,1]
    L_vis = ((l.squeeze(0).cpu() + 1.0) * 0.5).clamp(0, 1)

    # Plot + save
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input (L)")
    plt.imshow(L_vis, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted RGB")
    plt.imshow(pred_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Real RGB")
    plt.imshow(real_rgb)
    plt.axis("off")

    output_path = os.path.join(output_dir, f"result_{idx}.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

def as_load_img_array(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.tile(img[:, :, None], 3)

    # match Image.open -> np.asarray dtype/range behavior (uint8 0..255)
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).round().astype(np.uint8)

    return img


# show_results4(device, colorizer_eccv16, colorizer_siggraph17, testData, idx=i, size=size)
def show_results4(device, eccv16: ECCVGenerator, siggraph17: SIGGRAPHGenerator, data, idx, size, output_dir=CONFIG.RESULTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    eccv16.eval()
    siggraph17.eval()

    l, ab = data[idx]

    Lb = l.unsqueeze(0).to(device)  

    real_rgb = lab_to_rgb(Lb.cpu(), ab.unsqueeze(0).cpu())[0]
    real_rgb_u8 = as_load_img_array(real_rgb)
    (tens_l_orig, tens_l_rs) = preprocess_img(real_rgb_u8, HW=(256,256))

    with torch.no_grad():
        pred_ab_eccv16 = eccv16(tens_l_rs.to(device))
        pred_ab_siggraph17 = siggraph17(tens_l_rs.to(device))

    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, pred_ab_eccv16.cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, pred_ab_siggraph17.cpu())

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.title("Input (L)")
    plt.imshow(img_bw, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Predicted RGB eccv16")
    plt.imshow(out_img_eccv16)
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Predicted RGB siggraph17")
    plt.imshow(out_img_siggraph17)
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Real RGB")
    plt.imshow(real_rgb)
    plt.axis("off")

    output_path = os.path.join(output_dir, f"result_{idx}.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def visualize(model: torch.nn.Module, data, num_images: int = 5, output_dir="./results/midtrain"):

    os.makedirs(output_dir, exist_ok=True)

    model.generatorNet.eval()
    with torch.no_grad():
        model.setupInput(data)
        model.forward()
    model.generatorNet.train()
    fakeColor = model.fakeColor.detach()
    realColor = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fakeColor)
    real_imgs = lab_to_rgb(L, realColor)
    fig = plt.figure(figsize=(15, 8))
    for i in range(num_images):
        ax = plt.subplot(3, num_images, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, num_images, i + 1 + 2*num_images)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    output_path = os.path.join(output_dir, f"colorization_{time.time()}.png")
    fig.savefig(output_path)
    plt.close()

def _colorizeFromImage(device, model: torch.nn.Module, imagePath: str , size, output_dir="./results/examples"):
    os.makedirs(output_dir, exist_ok=True)

    img = PIL.Image.open(imagePath)
    img = img.resize((size, size))
    img = transforms.ToTensor()(img)[:1] * 2. - 1.

    model.generatorNet.eval()

    with torch.no_grad():
        # Predict ab

        pred_ab = model.generatorNet(img.unsqueeze(0).to(device))

    colorized = lab_to_rgb(img.unsqueeze(0), pred_ab.cpu())[0]

    L_vis = ((img.squeeze(0).cpu() + 1.0) * 0.5).clamp(0, 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input (L)")
    plt.imshow(L_vis, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted RGB")
    plt.imshow(colorized)
    plt.axis("off")

    output_path = os.path.join(output_dir, f"result_{time.time()}.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()