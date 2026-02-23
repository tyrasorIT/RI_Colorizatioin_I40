from typing import List
from utils.dataset import COCO_LAB
import torch
import torch.distributed as tdist
from datetime import datetime
from torch import nn
from models.ResNetUNetColorization import ResNetUNetColorization
from pretrainers.pretrainGenerator import pretrainGenerator
from utils.results import colorizeFromImage, show_results3, show_results4
from utils.model import saveModel, buildResNetFastAi, loadModelCheckpoint
from models.GANColorization import GANColorization
from external.colorization.colorizers import eccv16, siggraph17
from config import CONFIG
from trainers.GANTrainer import GANTrainer

def displaySelectedResults3(device, modelPath: str, testData, idxList: List[int], size, pretrainedGeneratorPath=None, generatorType=None):
    # Create model with pretrained=False since we're loading trained weights

    if pretrainedGeneratorPath is not None:
        if generatorType == "pretrainedResnet":
            generatorNet = ResNetUNetColorization(out_ch=2, pretrained=True)
        elif generatorType == "fastai":
            generatorNet = buildResNetFastAi(device, n_input=1, n_output=2, size=128)

        loadModelCheckpoint(device, pretrainedGeneratorPath, generatorNet)
    else:
        generatorNet = "resnet"

    model_wrapper = GANColorization(device, generatorNet=generatorNet)
    
    loadModelCheckpoint(device, modelPath, model_wrapper)
    model_wrapper.eval()
    model_wrapper.generatorNet.eval()

    # Explicitly set all submodules to eval
    for module in model_wrapper.generatorNet.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.track_running_stats = True
    
    for i in idxList:
        show_results3(device, model_wrapper, testData, idx=i, size=size)

def colorizeFromImage(device, modelPath: str, imagePath: str, size):
    model = GANColorization(device, generatorNet="resnet")

    loadModelCheckpoint(device, modelPath, model)

    colorizeFromImage(device, model, imagePath, size)

def displayIndustrialModelResults(device, testData, idxList: List[int], size):
    colorizer_eccv16 = eccv16(pretrained=True).to(device)
    colorizer_siggraph17 = siggraph17(pretrained=True).to(device)

    for i in idxList:
        show_results4(device, colorizer_eccv16, colorizer_siggraph17, testData, idx=i, size=size)

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    if CONFIG.MODE == CONFIG.RunMode.INIT:
        print("[MAIN] In init mode!")
        trainData = COCO_LAB(split="train")
        testData = COCO_LAB(split="val")
        print("[MAIN] Trainer initialized successfully!")
        print("[MAIN] Start training with [torchrun --nproc_per_node=<nGPU>]|python main.py train|infer <args>")
    else:
        print("[MAIN] In train mode!")

        if CONFIG.USE_DDP:
            tdist.init_process_group(backend="nccl")

        imSize=CONFIG.args.imSize
        trainData = COCO_LAB(size=imSize, split="train")
        testData = COCO_LAB(size=imSize, split="val")
        batchSize = CONFIG.args.batchSize
        numEpochs = CONFIG.args.numEpoch
        pretrainedGeneratorPath= CONFIG.args.pretrainGeneratorPath
        generatorType = CONFIG.args.generatorType
        formattedTime = datetime.now().strftime("%d%m%Y%H%M")
        modelPath = f"colorizer_{generatorType}_epoch{numEpochs}_{formattedTime}.pth"
        imgToDisplay = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        imgToDisplayTrain = [21, 23, 25, 27, 29, 31, 33, 35, 37, 39]
        fullRangeToDisplay = imgToDisplay + imgToDisplayTrain

        # mode = 2 #1 is train, 0 is show #2 is new train #3 is new show

        if CONFIG.MODE == CONFIG.RunMode.TRAIN:
            model = GANTrainer(CONFIG.DEVICE, trainData, testData, batchSize = batchSize, numEpochs = numEpochs, displayEvery=100, pretrainedGeneratorPath=pretrainedGeneratorPath, generatorType=generatorType)
            if CONFIG.LOCAL_RANK == 0:
                saveModel(modelPath, model.module, numEpochs) 
                show_results3(CONFIG.DEVICE, model.module, testData, idx = 1, size=imSize)
        elif CONFIG.MODE == CONFIG.RunMode.PRETRAIN:
            model = pretrainGenerator(CONFIG.DEVICE, generatorType, trainData, batchSize=batchSize, numEpochs=numEpochs)
            if CONFIG.LOCAL_RANK == 0:
                #Check input model and generate name base on it
                # saveModel("resnet_unet_pretrained.pt", model.module, numEpochs)
                saveModel("resnet_fastai_unet_pretrained.pt", model.module, numEpochs)
        elif CONFIG.MODE == CONFIG.RunMode.INFER:
            if CONFIG.LOCAL_RANK == 0:
                displaySelectedResults3(CONFIG.DEVICE, modelPath, testData, fullRangeToDisplay, size=imSize, pretrainedGeneratorPath=pretrainedGeneratorPath, generatorType=generatorType)

            # elif mode == 4:
            #     if CONFIG.LOCAL_RANK == 0:
            #         imagePath = "examples/example1_bw.jpeg"
            #         colorizeFromImage(CONFIG.DEVICE, modelPath, imagePath, size=imSize)

            # elif mode == 8:
            #     if CONFIG.LOCAL_RANK == 0:
            #         displayIndustrialModelResults(CONFIG.DEVICE, testData, fullRangeToDisplay, imSize)

        if CONFIG.USE_DDP:
            tdist.destroy_process_group()
    
