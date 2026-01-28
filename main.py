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
from utils.results import show_results
from customPerceptualLoss import PerceptualLoss, PerceptualLoss2
from customColorizationLoss import CombinedColorizationLoss, CombinedColorizationLossCIEDE
from customChromaLoss import ChromaLoss
from utils.model import saveModel
from torch.cuda.amp import GradScaler

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

def loadModelAndDisplaySelectedResults(device, modelPath: str, testData, idxList: List[int], size):
    # model = UNetColorization(in_ch=1, out_ch=2, base_ch=32).to(device)
    model = ResNetUNetColorization(out_ch=2, pretrained=True).to(device)

    checkpoint = torch.load(modelPath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    for i in idxList:
        show_results(device, model, testData, idx=i, size=size)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cudnn.benchmark = True

    tdist.init_process_group(backend="nccl")

    imSize=64
    # trainData = STL10_LAB(split="train", size=imSize)
    # testData = STL10_LAB(split="test", size=imSize)
    trainData = COCO_LAB(root="./data/coco_set/train2017/train2017/", size=imSize)
    testData = COCO_LAB(root="./data/coco_set/val2017/val2017/", size=imSize)
    batchSize = 384
    numEpochs = 50
    modelPath = "colorizer_unet_epoch50.pth"
    imgToDisplay = [0, 1, 3, 5, 9]

    mode = 0#1 is train, 0 is show

    if mode:
        model = mainTrainer(device, trainData, testData, batchSize = batchSize, numEpochs = numEpochs)
        if local_rank == 0:
            saveModel(modelPath, model.module, numEpochs) 
            show_results(device, model, testData, idx = 1, size=imSize)
    else:
        if local_rank == 0:
            loadModelAndDisplaySelectedResults(device, modelPath, testData, imgToDisplay, size=imSize)

    tdist.destroy_process_group()
    
