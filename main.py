import torch
import torch.distributed as tdist
import os
import time
from typing import List
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import optim
from datasetCreator import STL10_LAB, COCO_LAB
from customAutoencoder import Autoencoder, UNetAutoencoder, UNetColorization, ResNetUNetColorization
from modelLoops.testLoop import test_loop
from modelLoops.trainLoop import train_loop
from utils.results import show_results, show_results2, visualize, colorizeFromImage, show_results3, show_results4
from customPerceptualLoss import PerceptualLoss, PerceptualLoss2
from customColorizationLoss import CombinedColorizationLoss, CombinedColorizationLossCIEDE
from customChromaLoss import ChromaLoss
from utils.model import saveModel, buildResNetFastAi
from torch.cuda.amp import GradScaler
from models.GANColorization import GANColorization
from utils.lossTrack import create_loss_meters, update_losses, log_results
from tqdm import tqdm
from colorization.colorizers import eccv16, siggraph17

local_rank = int(os.environ["LOCAL_RANK"])

def reduce_mean(tensor):
    tdist.all_reduce(tensor, op=tdist.ReduceOp.SUM)
    tensor /= tdist.get_world_size()
    return tensor

def mainTrainer(device, trainData, testData, batchSize = 64, numEpochs = 50):
    # batch size and loaders
    trainSampler = DistributedSampler(trainData)
    testSampler = DistributedSampler(testData, shuffle=False)

    trainLoader = DataLoader(trainData, batch_size=batchSize, sampler=trainSampler, num_workers=8, pin_memory=True, persistent_workers=True)
    testLoader = DataLoader(testData, batch_size=batchSize, sampler=testSampler, num_workers=8, pin_memory=True, persistent_workers=True)

    # model = Autoencoder().to(device)
    # model = UNetAutoencoder().to(device)
    # model = UNetColorization(in_ch=1, out_ch=2, base_ch=32).to(device)
    #model = ResNetUNetColorization(pretrained=True).to(device)
    model = ResNetUNetColorization(pretrained=True).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank]
    )
    pixelLossFn = nn.SmoothL1Loss()
    percepLossFn = PerceptualLoss().to(device)
    chromaLossFn = ChromaLoss().to(device)
    # percepLossFn = PerceptualLoss2().to(device)
    # lossFn = nn.L1Loss()
    lossFn = CombinedColorizationLoss(pixelLossFn, percepLossFn, chromaLossFn, pixel_weight=0.5, percep_weight=0.2, chroma_weight=0.2).to(device)
    # lossFn = CombinedColorizationLossCIEDE(percepLossFn, percep_weight=0.5).to(device)
    learRate = 6e-4
    weightDecay = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learRate, weight_decay=weightDecay)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # we want to minimize validation loss
        factor=0.5,      # multiply lr by this when plateau
        patience=3,      # epochs with no improvement before reducing
        verbose=True,
        min_lr=1e-6      # don’t go below this
    )
    epochTimes = []

    for epoch in range(numEpochs):
        start = time.time()
        if local_rank == 0:
            print(f"Epoch {epoch+1}: ")
        trainSampler.set_epoch(epoch)
        train_loop(model, trainLoader , lossFn, optimizer, device, scaler)
        testLoss = test_loop(model, testLoader, lossFn, device)

        testLoss = torch.tensor(testLoss, device=device)
        testLoss = reduce_mean(testLoss)
        testLoss = testLoss.item()
        if local_rank == 0:
            scheduler.step(testLoss)

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

    return model

def mainTrainer2(device, trainData, testData, batchSize = 64, numEpochs = 50, displayEvery=200, pretrainedGeneratorPath=None, generatorType=None):
    # batch size and loaders
    trainSampler = DistributedSampler(trainData)
    testSampler = DistributedSampler(testData, shuffle=False)

    trainLoader = DataLoader(trainData, batch_size=batchSize, sampler=trainSampler, num_workers=8, pin_memory=True, persistent_workers=True, shuffle=False)
    testLoader = DataLoader(testData, batch_size=batchSize, sampler=testSampler, num_workers=8, pin_memory=True, persistent_workers=True)

    # model = GANColorization(device)
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

def loadModelAndDisplaySelectedResults(device, modelPath: str, testData, idxList: List[int], size):
    # model = UNetColorization(in_ch=1, out_ch=2, base_ch=32).to(device)
    model = ResNetUNetColorization(out_ch=2, pretrained=True).to(device)

    checkpoint = torch.load(modelPath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    for i in idxList:
        show_results(device, model, testData, idx=i, size=size)

def loadModelAndDisplaySelectedResults2(device, modelPath: str, testData, idxList: List[int], size):
    # model = UNetColorization(in_ch=1, out_ch=2, base_ch=32).to(device)
    model = GANColorization(device , generatorNet="resnet")

    checkpoint = torch.load(modelPath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    for i in idxList:
        show_results2(device, model, testData, idx=i, size=size)

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

def loadModelAndDisplaySelectedResults_TrainingStyle(device, modelPath: str, testData, idxList: List[int], size):
    generator = ResNetUNetColorization(out_ch=2, pretrained=False).to(device)
    model = GANColorization(device, generatorNet=generator)
    
    checkpoint = torch.load(modelPath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.generatorNet.eval()
    
    # Use DataLoader exactly like training
    testLoader = DataLoader(testData, batch_size=16, shuffle=False)  # Same batch size as training viz
    
    # Get a batch and use visualize() - the SAME function that worked during training
    test_batch = next(iter(testLoader))
    visualize(device, model, test_batch, output_dir="./results/loaded_model_viz")

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
    # trainData = STL10_LAB(split="train", size=imSize)
    # testData = STL10_LAB(split="test", size=imSize)
    trainData = COCO_LAB(root="./data/coco_set/train2017/train2017/", size=imSize, split="train")
    testData = COCO_LAB(root="./data/coco_set/val2017/val2017/", size=imSize, split="val")
    batchSize = 256
    numEpochs = 100
    pretrainedGeneratorPath= "resnet_fastai_unet_pretrained.pt"
    generatorType = "fastai"
    modelPath = f"colorizer_{generatorType}_epoch{numEpochs}.pth"
    imgToDisplay = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    imgToDisplayTrain = [21, 23, 25, 27, 29, 31, 33, 35, 37, 39]
    fullRangeToDisplay = imgToDisplay + imgToDisplayTrain

    mode = 8 #1 is train, 0 is show #2 is new train #3 is new show

    if mode == 1:
        model = mainTrainer(device, trainData, testData, batchSize = batchSize, numEpochs = numEpochs)
        if local_rank == 0:
            saveModel(modelPath, model.module, numEpochs) 
            show_results(device, model, testData, idx = 1, size=imSize)
    elif mode == 2:
        model = mainTrainer2(device, trainData, testData, batchSize = batchSize, numEpochs = numEpochs, displayEvery=100, pretrainedGeneratorPath=pretrainedGeneratorPath, generatorType=generatorType)
        if local_rank == 0:
            saveModel(modelPath, model.module, numEpochs) 
            show_results2(device, model.module, testData, idx = 1, size=imSize)
    elif mode == 0:
        if local_rank == 0:
            loadModelAndDisplaySelectedResults(device, modelPath, testData, imgToDisplay, size=imSize)
    elif mode == 3:
        if local_rank == 0:
            # loadModelAndDisplaySelectedResults3(device, modelPath, testData, imgToDisplay, size=imSize, pretrainedGeneratorPath=pretrainedGeneratorPath)
            loadModelAndDisplaySelectedResults3(device, modelPath, testData, fullRangeToDisplay, size=imSize, pretrainedGeneratorPath=pretrainedGeneratorPath, generatorType=generatorType)
    elif mode == 4:
        if local_rank == 0:
            imagePath = "examples/example1_bw.jpeg"
            loadModelAndSaveFromImage(device, modelPath, imagePath, size=imSize)
    elif mode == 5:
        if local_rank == 0:
            loadModelAndDisplaySelectedResults_TrainingStyle(device, modelPath, testData, imgToDisplay, imSize)
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
    
