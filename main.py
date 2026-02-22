import os
import time
from typing import List
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.dataset import COCO_LAB
import torch
import torch.distributed as tdist
from torch import nn
from torch import optim
from models.ResNetUNetColorization import ResNetUNetColorization
from utils.results import show_results, show_results2, visualize, colorizeFromImage, show_results3, show_results4
from utils.model import saveModel, buildResNetFastAi
from models.GANColorization import GANColorization
from utils.lossTrack import create_loss_meters, update_losses, log_results
from tqdm import tqdm
from colorization.colorizers import eccv16, siggraph17

local_rank = int(os.environ["LOCAL_RANK"])

def mainTrainer2(device, trainData, testData, batchSize = 64, numEpochs = 50, displayEvery=200, pretrainedGeneratorPath=None, generatorType=None):
    # batch size and loaders
    trainSampler = DistributedSampler(trainData)
    testSampler = DistributedSampler(testData, shuffle=False)

    trainLoader = DataLoader(trainData, batch_size=batchSize, sampler=trainSampler, num_workers=8, pin_memory=True, persistent_workers=True, shuffle=False)
    testLoader = DataLoader(testData, batch_size=batchSize, sampler=testSampler, num_workers=8, pin_memory=True, persistent_workers=True)

    if pretrainedGeneratorPath is not None:
        if generatorType == "customResnet":
            generatorNet = ResNetUNetColorization(out_ch=2, pretrained=True)
            checkpoint = torch.load(pretrainedGeneratorPath, map_location=device)
            generatorNet.load_state_dict(checkpoint["model_state_dict"])
        elif generatorType == "fastai":
            generatorNet = buildResNetFastAi(device, n_input=1, n_output=2, size=128)
            checkpoint = torch.load(pretrainedGeneratorPath, map_location=device)
            generatorNet.load_state_dict(checkpoint["model_state_dict"])
    else:
        generatorNet = "resnet"
    model = GANColorization(device, generatorNet=generatorNet)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank]
    )

    epochTimes = []

    for epoch in range(numEpochs):
        trainSampler.set_epoch(epoch)
        loss_meter_dict = create_loss_meters()
        i = 0
        start = time.time()
        if local_rank == 0:
            print(f"Epoch {epoch+1}: ")

        for data in tqdm(trainLoader, disable=(local_rank != 0), dynamic_ncols=True):
            insideModel = model.module
            insideModel.setupInput(data)
            insideModel.optimize()
            update_losses(insideModel, loss_meter_dict, count=data[0].size(0))
            i +=1
            if local_rank == 0 and (i % displayEvery == 0):
                print(f"\nEpoch {epoch+1}/{numEpochs}")
                print(f"Iteration {i}/{len(trainLoader)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(device, insideModel, data, 6, output_dir="./results/midtrain2")

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
        if local_rank == 0:
            print(f"Epoch took {epochDuration:.2f} sec")
            print(f"Estimated time remaining: {int(eta_h)}h {int(eta_m)}m {eta_s}s")
            print(f"\nEpoch {epoch+1}/{numEpochs}")
            print(f"Iteration {i}/{len(trainLoader)}")
            log_results(loss_meter_dict) # function to print out the losses

    return model

def loadModelAndDisplaySelectedResults3(device, modelPath: str, testData, idxList: List[int], size, pretrainedGeneratorPath=None, generatorType=None):
    # Load checkpoint first to check what we have
    mainModelCheckpoint = torch.load(modelPath, map_location=device)
    
    # Create model with pretrained=False since we're loading trained weights

    if pretrainedGeneratorPath is not None:
        if generatorType == "customResnet":
            generatorNet = ResNetUNetColorization(out_ch=2, pretrained=True)
            checkpoint = torch.load(pretrainedGeneratorPath, map_location=device)
            generatorNet.load_state_dict(checkpoint["model_state_dict"])
        elif generatorType == "fastai":
            generatorNet = buildResNetFastAi(device, n_input=1, n_output=2, size=128)
            checkpoint = torch.load(pretrainedGeneratorPath, map_location=device)
            generatorNet.load_state_dict(checkpoint["model_state_dict"])
    else:
        generatorNet = "resnet"

    model_wrapper = GANColorization(device, generatorNet=generatorNet)
    
    # Now load the trained weights
    model_wrapper.load_state_dict(mainModelCheckpoint["model_state_dict"])
    model_wrapper.eval()
    model_wrapper.generatorNet.eval()

    # Explicitly set all submodules to eval
    for module in model_wrapper.generatorNet.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.track_running_stats = True
    
    for i in idxList:
        show_results3(device, model_wrapper, testData, idx=i, size=size)

def loadModelAndSaveFromImage(device, modelPath: str, imagePath: str, size):
    model = GANColorization(device, generatorNet="resnet")

    checkpoint = torch.load(modelPath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    colorizeFromImage(device, model, imagePath, size)

def loadIndustrialModelAndDisplaySelectedResults(device, testData, idxList: List[int], size):
    colorizer_eccv16 = eccv16(pretrained=True).to(device)
    colorizer_siggraph17 = siggraph17(pretrained=True).to(device)

    for i in idxList:
        show_results4(device, colorizer_eccv16, colorizer_siggraph17, testData, idx=i, size=size)


def pretrain_generator(device, trainData, batchSize=128, numEpochs=20):
    # Create DataLoader
    trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=8)
    
    # Use your existing ResNet U-Net
    generator = ResNetUNetColorization(out_ch=2, pretrained=True).to(device)
    
    generator = torch.nn.parallel.DistributedDataParallel(
        generator, device_ids=[local_rank]
    )

    optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    
    generator.train()

    epochTimes = []
    
    for epoch in range(numEpochs):
        epoch_loss = 0
        num_batches = 0
        
        if local_rank == 0:
            start = time.time()
            print(f"Epoch {epoch+1}: ")
        
        for data in tqdm(trainLoader, disable=(local_rank != 0), dynamic_ncols=True):
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
        
        if local_rank == 0:
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

def pretrain_generatorFastAi(device, trainData, batchSize=128, numEpochs=20):
    # Create DataLoader
    trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=8)
    
    # Use your existing ResNet U-Net
    generator = buildResNetFastAi(device, n_input=1, n_output=2, size=128)
    
    generator = torch.nn.parallel.DistributedDataParallel(
        generator, device_ids=[local_rank]
    )

    optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    
    generator.train()

    epochTimes = []
    
    for epoch in range(numEpochs):
        epoch_loss = 0
        num_batches = 0
        
        if local_rank == 0:
            start = time.time()
            print(f"Epoch {epoch+1}: ")
        
        for data in tqdm(trainLoader, disable=(local_rank != 0), dynamic_ncols=True):
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
        
        if local_rank == 0:
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

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cudnn.benchmark = True

    tdist.init_process_group(backend="nccl")

    imSize=256
    trainData = COCO_LAB(size=imSize, split="train")
    testData = COCO_LAB(size=imSize, split="val")
    batchSize = 256
    numEpochs = 100
    pretrainedGeneratorPath= "resnet_fastai_unet_pretrained.pt"
    generatorType = "fastai"
    modelPath = f"colorizer_{generatorType}_epoch{numEpochs}.pth"
    imgToDisplay = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    imgToDisplayTrain = [21, 23, 25, 27, 29, 31, 33, 35, 37, 39]
    fullRangeToDisplay = imgToDisplay + imgToDisplayTrain

    mode = 2 #1 is train, 0 is show #2 is new train #3 is new show

    if mode == 1:
        pass
    elif mode == 2:
        model = mainTrainer2(device, trainData, testData, batchSize = batchSize, numEpochs = numEpochs, displayEvery=100, pretrainedGeneratorPath=pretrainedGeneratorPath, generatorType=generatorType)
        if local_rank == 0:
            saveModel(modelPath, model.module, numEpochs) 
            show_results2(device, model.module, testData, idx = 1, size=imSize)
    elif mode == 0:
        pass
    elif mode == 3:
        if local_rank == 0:
            loadModelAndDisplaySelectedResults3(device, modelPath, testData, fullRangeToDisplay, size=imSize, pretrainedGeneratorPath=pretrainedGeneratorPath, generatorType=generatorType)
    elif mode == 4:
        if local_rank == 0:
            imagePath = "examples/example1_bw.jpeg"
            loadModelAndSaveFromImage(device, modelPath, imagePath, size=imSize)
    elif mode == 5:
        pass
    elif mode == 6:
        model = pretrain_generator(device, trainData, batchSize=384, numEpochs=20)
        if local_rank == 0:
            saveModel("resnet_unet_pretrained.pt", model.module, numEpochs)
    elif mode == 7:
        model = pretrain_generatorFastAi(device, trainData, batchSize=256, numEpochs=20)
        if local_rank == 0:
            saveModel("resnet_fastai_unet_pretrained.pt", model.module, numEpochs)
    elif mode == 8:
        if local_rank == 0:
            loadIndustrialModelAndDisplaySelectedResults(device, testData, fullRangeToDisplay, imSize)

    tdist.destroy_process_group()
    
