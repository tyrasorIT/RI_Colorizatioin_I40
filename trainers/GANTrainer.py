from config import CONFIG
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from models.ResNetUNetColorization import ResNetUNetColorization
from models.GANColorization import GANColorization
from utils.model import buildResNetFastAi, loadModelCheckpoint
import torch
from utils.lossTrack import create_loss_meters, update_losses, log_results, CSVLogger, reduce_meter_dict
import time
import os
from datetime import datetime
from tqdm import tqdm
from utils.results import visualize
from config import CONFIG

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
        folderTime = datetime.now().strftime("%d%m%Y%H%M")
        outputDir = f"./results/run_{folderTime}"
        outputVizDir = f"{outputDir}/midtrain_viz"
        os.makedirs(outputDir, exist_ok=True)

        csvPath = os.path.join(outputDir, "metrics.csv")

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

        with open(os.path.join(outputDir, "experiment_info.txt"), "w") as f:
            f.write(f"Generator Type: {generatorType}\n")
            f.write(f"Batch Size: {batchSize}\n")
            f.write(f"Epochs: {numEpochs}\n")
            f.write(f"Device: {torch.cuda.get_device_name(0)}\n")
            f.write(f"GPU Count: {torch.cuda.device_count()}\n")
            f.write(f"Torch Version: {torch.__version__}\n")
            f.write(f"CUDA Version: {torch.version.cuda}\n")

    epochTimes = []
    best_val_loss = float("inf")

    for epoch in range(numEpochs):
        if CONFIG.USE_DDP:
            trainSampler.set_epoch(epoch)
            testSampler.set_epoch(epoch)

        model.train()
        train_meters = create_loss_meters()
        
        # i = 0
        start = time.time()
        if CONFIG.LOCAL_RANK == 0:
            print(f"Epoch {epoch+1}: ")

        for data in tqdm(trainLoader, disable=(CONFIG.LOCAL_RANK != 0), dynamic_ncols=True, desc="Training"):
            insideModel.setupInput(data)
            insideModel.optimize()
            insideModel.compute_metrics()
            update_losses(insideModel, train_meters, count=data[0].size(0))
            # i +=1
            # if CONFIG.LOCAL_RANK == 0 and (i % displayEvery == 0):
            #     print(f"\nEpoch {epoch+1}/{numEpochs}")
            #     print(f"Iteration {i}/{len(trainLoader)}")
            #     log_results(train_meters) # function to print out the losses
        visualize(insideModel, data, 6, output_dir=outputVizDir)

        if CONFIG.USE_DDP:
            reduce_meter_dict(train_meters, device)

        model.eval()
        val_meters = create_loss_meters()

        with torch.no_grad():
            for data in tqdm(testLoader, disable=(CONFIG.LOCAL_RANK != 0), dynamic_ncols=True, desc="Validation"):
                insideModel.setupInput(data)
                insideModel.forward()
                insideModel.backwardDiscriminator()
                insideModel.backwardGenerator()
                insideModel.compute_metrics()
                update_losses(insideModel, val_meters, count=data[0].size(0))

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

            print(f"\nEpoch {epoch+1}/{numEpochs}")
            print(f"Train G: {train_G:.4f} | Train D: {train_D:.4f}")
            print(f"Val   G: {val_G:.4f} | Val   D: {val_D:.4f}")
            print(f"PSNR: {val_PSNR:.4f} | SSIM: {val_SSIM:.4f}")
            print(f"Epoch time: {epochDuration:.2f}s")
            print(f"Estimated time remaining: {int(eta_h)}h {int(eta_m)}m {eta_s}s")
            # log_results(train_meters) # function to print out the losses

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

    return model
