import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from skimage import color
import os

def show_results(device, model: torch.nn.Module, data, idx, size, output_dir="./results"):
    model.eval()
    l, ab = data[idx]

    with torch.no_grad():
        # Predict ab
        os.makedirs(output_dir, exist_ok=True)

        predAB = model(l.unsqueeze(0).to(device))[0].cpu()

        # Undo normalization
        imgL = (l.squeeze(0).numpy() * 100.0)
        imgPredAB = predAB.clone()
        imgPredAB[0] *= 110.0
        imgPredAB[1] *= 110.0
        imgPredAB = imgPredAB.permute(1, 2, 0).numpy()

        imgAB = ab.clone()
        imgAB[0] *= 110.0
        imgAB[1] *= 110.0
        imgAB = imgAB.permute(1, 2, 0).numpy()

        # Convert LAB → RGB
        realLAB = np.zeros((size,size,3))
        labPredAB = np.zeros((size,size,3))
        realLAB[:,:,0] = imgL
        realLAB[:,:,1:] = imgAB
        labPredAB[:,:,1:] = imgPredAB
        labPredAB[:,:,0] = imgL
        realRGB = color.lab2rgb(realLAB)
        predRGB = color.lab2rgb(labPredAB)

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.title("Black & white")
        plt.imshow(imgL, cmap="gray")
        plt.axis("off")
        plt.subplot(1,3,2)
        plt.title("Predicted RGB")
        plt.imshow(predRGB)
        plt.axis("off")
        plt.subplot(1,3,3)
        plt.title("Real RGB")
        plt.imshow(realRGB)
        plt.axis("off")

        output_path = os.path.join(output_dir, f"result_{idx}.png")
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()

        #plt.show()