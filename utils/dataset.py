from torchvision import transforms
from torch.utils.data import Dataset
from skimage.color import rgb2lab
from PIL import Image
import numpy as np
import os
import requests
from tqdm import tqdm
from pathlib import Path
import zipfile
from datetime import datetime

COCO_URLS = {
    "train": "http://images.cocodataset.org/zips/train2017.zip",
    "val": "http://images.cocodataset.org/zips/val2017.zip",
}

def _get_project_root():
    return Path(__file__).resolve().parent.parent

def _downloadFileResumable(url: str, destPath):
    tempPath = destPath.with_suffix(destPath.suffix + ".part")

    response = requests.head(url)
    totalSize = int(response.headers.get('content-length', 0))

    if os.path.exists(tempPath):
        downloadedSize = os.path.getsize(tempPath)
    else:
        downloadedSize = 0

    if downloadedSize >= totalSize and totalSize > 0:
        os.rename(tempPath, destPath)
        print(f"[Downloader] Download already complete: {destPath}")
        return
    
    headers = {
        "Range": f"bytes={downloadedSize}"
    }

    response = requests.get(url, headers=headers, stream=True)

    mode = "ab" if downloadedSize > 0 else "wb"

    with open(tempPath, mode) as file, tqdm(
        total=totalSize,
        initial=downloadedSize,
        unit='B',
        unit_scale=True,
        desc=os.path.basename(destPath)
    ) as bar:
        
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))


    os.rename(tempPath, destPath)
    print(f"[Downloader] Download Finished: {destPath}")
    
def _downloadCOCOBySplit(split: str = "train"):
    projectRoot = _get_project_root()
    root = projectRoot / "data" / "coco"
    root = Path(root)

    root.mkdir(parents=True, exist_ok=True)

    url = COCO_URLS[split]
    zipName = f"{split}2017.zip"
    folderName = f"{split}2017"

    zipPath = root / zipName
    extractPath = root / folderName

    if extractPath.exists():
        print(f"[COCO] {split} already available")
        return str(extractPath)
    
    print(f"[COCO] Downloading {split}2017...")
    _downloadFileResumable(url, zipPath)

    print(f"[COCO] Extracting {split}2017...")
    # with zipfile.ZipFile(zipPath, 'r') as zipRef:
    #     zipRef.extractall(root)
    with zipfile.ZipFile(zipPath, 'r') as zipRef:
        members = zipRef.infolist()
        total_size = sum(m.file_size for m in members)

        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Extracting") as pbar:
            for member in members:
                zipRef.extract(member, root)
                pbar.update(member.file_size)

    print(f"[COCO] Done")
    return str(extractPath)

class COCO_LAB(Dataset):
    """
    COCO dataset loader for LAB colorization.
    Loads raw images only (no annotations).
    """
    def __init__(self, size: int = 128, split="train"):
        self.size = size
        self.split = split

        root = _downloadCOCOBySplit(split=self.split)
        
        if not os.path.exists(root):
            raise RuntimeError(f"[COCO] Dataset path does not exist: {root}")

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip()
            ])
        elif split == "val":
            self.transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            ])

        # Load all image files
        self.image_paths = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(self.image_paths) == 0:
            raise RuntimeError(f"[COCO] No images found in {root}")

        print(f"[COCO] Loaded {len(self.image_paths)} images from {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)

        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1

        return L, ab
    
class DatasetValidator:
    def __init__(self, basePath: str):
        self.basePath = Path(basePath)

    def checkExists(self):
        trainPath = os.path.join(self.basePath, "train2017")
        valPath = os.path.join(self.basePath, "val2017")

        trainExists = os.path.exists(trainPath)
        valExists = os.path.exists(valPath)

        if not trainExists or not valExists:
            return False
        
        self.trainPath = trainPath
        self.valPath = valPath

        return True
    
    def getDatasetInfo(self):

        if not self.basePath.exists():
            print("[DATASET] Dataset base does not exist. Initialization required.")
            return {}
        
        trainSize = self.countImages(self.trainPath)
        valSize = self.countImages(self.valPath)

        if trainSize == 0 or valSize == 0:
            print("[DATASET] Dataset folders empty. Initialization required.")
            return {}
        
        lastVerified = datetime.now().isoformat()
        
        return {
            "trainSize": trainSize,
            "valSize": valSize,
            "lastVerified": lastVerified
        }
        
    def countImages(self, folder):
        if not os.path.exists(folder):
            return 0
        
        return len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        

#Local test
if __name__ == "__main__":
    imSize = 128

    trainData = COCO_LAB(size=imSize, split="train")
    testData = COCO_LAB(size=imSize, split="val")