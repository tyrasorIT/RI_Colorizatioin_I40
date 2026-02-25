import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import optim
from models.ResNetUNetColorization import ResNetUNetColorization
from utils.model import buildResNetFastAi
from utils.model import countParameters
from tqdm import tqdm
import os
from config import CONFIG
import shutil

def pretrainGenerator(device, generatorType: str, trainData, batchSize=128, numEpochs=20):
    # Create DataLoader
    trainSampler = DistributedSampler(trainData) if CONFIG.USE_DDP else None
    trainLoader = DataLoader(
        trainData,
        batch_size=batchSize,
        sampler=trainSampler,
        shuffle=False if CONFIG.USE_DDP else True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    
    if generatorType == CONFIG.GeneratorTypes.RESNET:
        generator = ResNetUNetColorization(out_ch=2, pretrained=True).to(device)
    elif generatorType == CONFIG.GeneratorTypes.FASTAI:
        generator = buildResNetFastAi(device, n_input=1, n_output=2, size=128)
    else:
        raise RuntimeError("[PRETRAINER] No generator selected! Exiting!")

    if CONFIG.USE_DDP:
        generator = torch.nn.parallel.DistributedDataParallel(
            generator, device_ids=[CONFIG.LOCAL_RANK]
        )

    optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    criterion = torch.nn.L1Loss()  

    epochTimes = []
    minLoss = float("inf")

    generator.train()
    
    for epoch in range(numEpochs):
        if CONFIG.USE_DDP:
            trainSampler.set_epoch(epoch)

        epoch_loss = 0
        num_batches = 0
        
        if CONFIG.LOCAL_RANK == 0:
            start = time.time()
            print(f"\nEpoch {epoch+1}/{numEpochs}: ")
        
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
            minLoss = avg_loss
            print(f"L1 Loss: {avg_loss:.5f}")
            print(f"Epoch took {epochDuration:.2f} sec")
            print(f"Estimated time remaining: {int(eta_h)}h {int(eta_m)}m {eta_s}s")

    if CONFIG.LOCAL_RANK == 0:
        os.makedirs(CONFIG.RUN_DIR, exist_ok=True)
        os.makedirs(CONFIG.MODEL_DIR, exist_ok=True)
        with open(os.path.join(CONFIG.RUN_DIR, "experiment_info.txt"), "a") as f:
            f.write(f"Mode: {CONFIG.MODE.value}\n")
            f.write(f"Generator Type: {generatorType.value}\n")
            if CONFIG.PRETRAINED:
                f.write(f"Pretrained generator used: {CONFIG.PRETRAINED_GENERATOR_PATH}\n")
            f.write(f"Image Size: {CONFIG.IMAGE_SIZE}\n")
            f.write(f"Batch Size: {batchSize}\n")
            f.write(f"Epochs: {numEpochs}\n")
            f.write(f"Device: {torch.cuda.get_device_name(0)}\n")
            f.write(f"GPU Count: {torch.cuda.device_count()}\n")
            f.write(f"Per GPU peak memory: {(torch.cuda.max_memory_allocated(CONFIG.LOCAL_RANK) / (1024 ** 2)):.1f} MB\n")
            f.write(f"Torch Version: {torch.__version__}\n")
            f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(f"LocalModelPath: {CONFIG.MODEL_PATH}\n")
            f.write(f"L1 Loss: {minLoss:.5f}")
            f.write(f"Total paremeters: {countParameters(generator, trainable_only=False)}\n")
            f.write(f"Trainable parameters: {countParameters(generator, trainable_only=True)}\n")
            f.write(f"Training time: {sum(epochTimes)}")

        shutil.copy(os.path.join(CONFIG.RUN_DIR, "experiment_info.txt"), CONFIG.MODEL_DIR)
    
    return generator