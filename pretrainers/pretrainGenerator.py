import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import optim
from models.ResNetUNetColorization import ResNetUNetColorization
from utils.model import buildResNetFastAi
from tqdm import tqdm
from config import CONFIG

def pretrainGenerator(device, generatorType: str, trainData, batchSize=128, numEpochs=20):
    # Create DataLoader
    trainSampler = DistributedSampler(trainData)
    trainLoader = DataLoader(
        trainData,
        batch_size=batchSize,
        sampler=trainSampler,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Use your existing ResNet U-Net
    if generatorType == "resnet":
        generator = ResNetUNetColorization(out_ch=2, pretrained=True).to(device)
    elif generatorType == "fastai":
        generator = buildResNetFastAi(device, n_input=1, n_output=2, size=128)
    else:
        RuntimeError("No generator selected! Exiting!")

    generator = torch.nn.parallel.DistributedDataParallel(
        generator, device_ids=[CONFIG.LOCAL_RANK]
    )

    optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    criterion = torch.nn.L1Loss()
    
    generator.train()

    epochTimes = []
    
    for epoch in range(numEpochs):
        trainSampler.set_epoch(epoch)

        epoch_loss = 0
        num_batches = 0
        
        if CONFIG.LOCAL_RANK == 0:
            start = time.time()
            print(f"Epoch {epoch+1}: ")
        
        for data in tqdm(trainLoader, disable=(CONFIG.LOCAL_RANK != 0), dynamic_ncols=True):
            L, ab = data  # Unpack tuple from COCO_LAB
            L, ab = L.to(device), ab.to(device)
            
            # Forward
            pred_ab = generator(L)
            loss = criterion(pred_ab, ab)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if CONFIG.LOCAL_RANK == 0:
            end = time.time()
            epochDuration = end - start
            epochTimes.append(epochDuration)

            # Estimate remaining time
            avgEpochTime = sum(epochTimes) / len(epochTimes)
            remainingEpochs = numEpochs - (epoch + 1)
            etaSeconds = avgEpochTime * remainingEpochs

            # Convert to readable time
            eta_h = etaSeconds // 3600
            eta_m = (etaSeconds % 3600) // 60
            eta_s = int(etaSeconds % 60)
            avg_loss = epoch_loss / num_batches
            print(f"Epoch took {epochDuration:.2f} sec")
            print(f"Estimated time remaining: {int(eta_h)}h {int(eta_m)}m {eta_s}s")
            print(f"Epoch {epoch+1}/{numEpochs}, L1 Loss: {avg_loss:.5f}")
    
    return generator