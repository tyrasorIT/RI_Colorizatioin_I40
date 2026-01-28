import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch import optim
from skimage import io, color

# Setup Autoencoder class
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
        )

    def forward(self, x):
        code = self.encoder(x)
        return self.decoder(code)

# Setup LAB conversion
def rgb_to_lab(img):
    img = img.permute(1, 2, 0).numpy()
    lab = color.rgb2lab(img).astype("float32")
    return lab

def split_and_normalize(lab):
    l = lab[:, :, 0] / 100.0
    ab = lab[:, :, 1:] / 128.0
    l = torch.tensor(l).unsqueeze(0)
    ab = torch.tensor(ab).permute(2, 0, 1)
    return l, ab

class STL10_LAB(Dataset):
    def __init__(self, split="train"):
        self.data = datasets.STL10(
            root="./data",
            split=split,
            download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, _ = self.data[idx]
        lab = rgb_to_lab(img)
        l, ab = split_and_normalize(lab)
        return l, ab

# batch size and loaders
batchSize = 64
trainData = STL10_LAB(split="train")
testData = STL10_LAB(split="test")
trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True)
testLoader = DataLoader(testData, batch_size=batchSize, shuffle=False)

def train_loop(model, loader, lossFn, optimizer, device):
    model.train()
    totalLoss = 0

    for l, ab in loader:
        x = l.to(device)
        y = ab.to(device)

        optimizer.zero_grad()

        predAB = model(x)
        predAB = torch.clamp(predAB, -1, 1)
        loss = lossFn(predAB, y)
        loss.backward()

        optimizer.step()

        totalLoss += loss.item()

    print("Total loss (train): ", totalLoss/len(loader))

def test_loop(model, loader, lossFn, device):
    model.eval()
    totalLoss = 0

    with torch.no_grad():
        for l, ab in loader:
            x = l.to(device)
            y = ab.to(device)

            predAB = model(x)
            predAB = torch.clamp(predAB, -1, 1)

            loss = lossFn(predAB, y)
            totalLoss += loss.item()

    print("Total loss (test): ", totalLoss/len(loader))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Autoencoder().to(device)
lossFn = nn.L1Loss()
learRate = 5e-4
optimizer = optim.Adam(model.parameters(), lr=learRate)
numEpochs = 300

for epoch in range(numEpochs):
    print("Epoch {}: ".format(epoch+1))
    train_loop(model, trainLoader , lossFn, optimizer, device)
    test_loop(model, testLoader, lossFn, device)

def show_results(model, data):
    model.eval()
    l, ab = data[0]

    with torch.no_grad():
        # Predict ab
        predAB = model(l.unsqueeze(0).to(device))[0].cpu()

        # Undo normalization
        imgL = (l.squeeze(0).numpy() * 100.0)
        imgPredAB = (predAB.permute(1, 2, 0) * 128.0)
        imgAB = (ab.permute(1, 2, 0).numpy() * 128.0)

        # Convert LAB → RGB
        realLAB = np.zeros((96,96,3))
        labPredAB = np.zeros((96,96,3))
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
        plt.show()

show_results(model, testData)
