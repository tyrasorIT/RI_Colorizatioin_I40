from config import CONFIG
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from models.GANColorization import GANColorization
import torch
from utils.lossTrack import create_loss_meters, update_losses, CSVLogger, reduce_meter_dict
import time
import os
from tqdm import tqdm
from utils.results import visualize
from utils.model import countParameters
from config import CONFIG
import shutil

def GANTrainer(device, trainData, testData, generatorType, batchSize = 64, numEpochs = 50, displayEvery=200):
    
    trainSampler = DistributedSampler(trainData) if CONFIG.USE_DDP else None
    testSampler = DistributedSampler(testData, shuffle=False) if CONFIG.USE_DDP else None

    trainLoader = DataLoader(trainData, batch_size=batchSize, sampler=trainSampler, num_workers=8, pin_memory=True, persistent_workers=True, shuffle=False if CONFIG.USE_DDP else True)
    testLoader = DataLoader(testData, batch_size=batchSize, sampler=testSampler, num_workers=8, pin_memory=True, persistent_workers=True)

    model = GANColorization(device, generatorType=generatorType, pretrained=CONFIG.PRETRAINED, pretrainedPath=CONFIG.PRETRAINED_GENERATOR_PATH)

    if CONFIG.USE_DDP:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[CONFIG.LOCAL_RANK]
        )

    insideModel = model.module if CONFIG.USE_DDP else model

    if CONFIG.LOCAL_RANK == 0:
        outputTrainVizDir = f"{CONFIG.RUN_DIR}/midtrain_viz"
        outputValVizDir = f"{CONFIG.RUN_DIR}/midval_viz"
        os.makedirs(CONFIG.RUN_DIR, exist_ok=True)

        csvPath = os.path.join(CONFIG.RUN_DIR, "metrics.csv")

        fieldnames = [
            "epoch",
            "train_lossGenerator",
            "train_lossDiscriminator",
            "val_lossGenerator",
            "val_lossDiscriminator",
            "val_PSNR",
            "val_SSIM",
            "epoch_time_sec"
        ]

        logger = CSVLogger(csvPath, fieldnames)

    epochTimes = []
    best_val_loss = float("inf")

    for epoch in range(numEpochs):
        if CONFIG.USE_DDP:
            trainSampler.set_epoch(epoch)

        model.train()
        train_meters = create_loss_meters()
        
        i = 0
        start = time.time()
        if CONFIG.LOCAL_RANK == 0:
            print(f"\nEpoch {epoch+1}/{numEpochs}")

        for data in tqdm(trainLoader, disable=(CONFIG.LOCAL_RANK != 0), dynamic_ncols=True, desc="Training"):
            insideModel.setupInput(data)
            insideModel.optimize()
            update_losses(insideModel, train_meters, count=data[0].size(0))
            i +=1
            if CONFIG.LOCAL_RANK == 0 and (i % displayEvery == 0):
                visualize(insideModel, data, 6, output_dir=outputTrainVizDir)

        if CONFIG.USE_DDP:
            reduce_meter_dict(train_meters, device)

        model.eval()
        val_meters = create_loss_meters()
        singleValViz = True
        with torch.no_grad():
            for data in tqdm(testLoader, disable=(CONFIG.LOCAL_RANK != 0), dynamic_ncols=True, desc="Validation"):
                insideModel.setupInput(data)
                insideModel.forward()
                insideModel.compute_metrics()
                update_losses(insideModel, val_meters, count=data[0].size(0))
                if CONFIG.LOCAL_RANK == 0 and singleValViz:
                    singleValViz = False
                    visualize(insideModel, data, 6, output_dir=outputValVizDir)
        if CONFIG.USE_DDP:
            reduce_meter_dict(val_meters, device)

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
        if CONFIG.LOCAL_RANK == 0:
            train_G = train_meters["lossGenerator"].avg
            train_D = train_meters["lossDiscriminator"].avg
            val_G = val_meters["lossGenerator"].avg
            val_D = val_meters["lossDiscriminator"].avg
            val_PSNR = val_meters["psnr"].avg
            val_SSIM = val_meters["ssim"].avg

            print(f"Train G: {train_G:.4f} | Train D: {train_D:.4f}")
            print(f"Val   G: {val_G:.4f} | Val   D: {val_D:.4f}")
            print(f"PSNR: {val_PSNR:.4f} | SSIM: {val_SSIM:.4f}")
            print(f"Epoch time: {epochDuration:.2f}s")
            print(f"Estimated time remaining: {int(eta_h)}h {int(eta_m)}m {eta_s}s")

            logger.log({
                "epoch": epoch + 1,
                "train_lossGenerator": train_G,
                "train_lossDiscriminator": train_D,
                "val_lossGenerator": val_G,
                "val_lossDiscriminator": val_D,
                "val_PSNR": val_PSNR,
                "val_SSIM": val_SSIM,
                "epoch_time_sec": epochDuration
            })

    if CONFIG.LOCAL_RANK == 0:
        logger.close()
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
            f.write(f"Total paremeters: {countParameters(model, trainable_only=False)}\n")
            f.write(f"Trainable parameters: {countParameters(model, trainable_only=True)}\n")
            fullTimeSeconds = sum(epochTimes)
            fullTime_H = fullTimeSeconds // 3600
            fullTime_m = (fullTimeSeconds % 3600) // 60
            fullTime_s = int(fullTimeSeconds % 60)
            f.write(f"Training time: {int(fullTime_H)}h {int(fullTime_m)}m {fullTime_s}s")

        shutil.copy(os.path.join(CONFIG.RUN_DIR, "experiment_info.txt"), CONFIG.MODEL_DIR)
        shutil.copy(os.path.join(CONFIG.RUN_DIR, "metrics.csv"), CONFIG.MODEL_DIR)

    return model
