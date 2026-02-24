from config import CONFIG
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from models.ResNetUNetColorization import ResNetUNetColorization
from models.GANColorization import GANColorization
from utils.model import buildResNetFastAi, loadModelCheckpoint
import torch
from utils.lossTrack import create_loss_meters, update_losses, log_results
import time
import os
from datetime import datetime
from tqdm import tqdm
from utils.results import visualize
from config import CONFIG

def GANTrainer(device, trainData, testData, batchSize = 64, numEpochs = 50, displayEvery=200, pretrainedGeneratorPath=None, generatorType=None):
    
    trainSampler = DistributedSampler(trainData) if CONFIG.USE_DDP else None
    testSampler = DistributedSampler(testData, shuffle=False) if CONFIG.USE_DDP else None

    trainLoader = DataLoader(trainData, batch_size=batchSize, sampler=trainSampler, num_workers=8, pin_memory=True, persistent_workers=True, shuffle=False if CONFIG.USE_DDP else True)
    testLoader = DataLoader(testData, batch_size=batchSize, sampler=testSampler, num_workers=8, pin_memory=True, persistent_workers=True)

    if pretrainedGeneratorPath is not None:
        if generatorType == CONFIG.GeneratorTypes.PRETRAINEDRESNET:
            generatorNet = ResNetUNetColorization(out_ch=2, pretrained=True)
        elif generatorType == CONFIG.GeneratorTypes.FASTAI:
            generatorNet = buildResNetFastAi(device, n_input=1, n_output=2, size=128)
        else:
            RuntimeError("No generator selected! Exiting!")

        loadModelCheckpoint(device, pretrainedGeneratorPath, generatorNet)
    else:
        generatorNet = CONFIG.GeneratorTypes.RESNET
    model = GANColorization(device, generatorNet=generatorNet)

    if CONFIG.USE_DDP:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[CONFIG.LOCAL_RANK]
        )

    epochTimes = []

    if CONFIG.LOCAL_RANK == 0:
        folderTime = datetime.now().strftime("%d%m%Y%H%M")
        outputDir = f"./results/midtrainViz_{folderTime}"
        os.makedirs(outputDir, exist_ok=True)

    for epoch in range(numEpochs):
        model.train()

        if CONFIG.USE_DDP:
            trainSampler.set_epoch(epoch)
            testSampler.set_epoch(epoch)

        loss_meter_dict = create_loss_meters()
        i = 0
        start = time.time()
        if CONFIG.LOCAL_RANK == 0:
            print(f"Epoch {epoch+1}: ")

        for data in tqdm(trainLoader, disable=(CONFIG.LOCAL_RANK != 0), dynamic_ncols=True):
            insideModel = model.module if CONFIG.USE_DDP else model
            insideModel.setupInput(data)
            insideModel.optimize()
            update_losses(insideModel, loss_meter_dict, count=data[0].size(0))
            i +=1
            if CONFIG.LOCAL_RANK == 0 and (i % displayEvery == 0):
                print(f"\nEpoch {epoch+1}/{numEpochs}")
                print(f"Iteration {i}/{len(trainLoader)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(insideModel, data, 6, output_dir=outputDir)

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
            print(f"Epoch took {epochDuration:.2f} sec")
            print(f"Estimated time remaining: {int(eta_h)}h {int(eta_m)}m {eta_s}s")
            print(f"\nEpoch {epoch+1}/{numEpochs}")
            print(f"Iteration {i}/{len(trainLoader)}")
            log_results(loss_meter_dict) # function to print out the losses

    return model
