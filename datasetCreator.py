from torchvision import datasets, transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset
from utils.image import rgb_to_lab, split_and_normalize
from PIL import Image
import os

class STL10_LAB(Dataset):
    def __init__(self, split="train", size=96):
        transform = transforms.Compose([
            transforms.Resize(
                (size, size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ])

        self.data = datasets.STL10(
            root="./data",
            split=split,
            download=True,
            # transform=transforms.ToTensor()
            transform=transform
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, _ = self.data[idx]
        lab = rgb_to_lab(img)
        l, ab = split_and_normalize(lab)
        return l, ab
    
class COCO_LAB(Dataset):
    """
    COCO dataset loader for LAB colorization.
    Loads raw images only (no annotations).
    """
    def __init__(self, root: str, size: int = 128):
        self.size = size

        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

        # Load all image files
        self.image_paths = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root}")

        print(f"[COCO_LAB] Loaded {len(self.image_paths)} images from {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)

        lab = rgb_to_lab(img)
        l, ab = split_and_normalize(lab)

        return l, ab