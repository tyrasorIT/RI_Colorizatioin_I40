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

def displaySelectedResults3(device, modelPath: str, testData, idxList: List[int], size, generatorType=None):

    metadata = loadModelCheckpoint(device, modelPath, None, metadataOnly=True)
    #If metadata exists use from metadata otherwise use from input
    if metadata:
        generatorType=CONFIG.GeneratorTypes(metadata["generator_type"])

    model_wrapper = GANColorization(device, generatorType=generatorType)
    
    loadModelCheckpoint(device, modelPath, model_wrapper)
    model_wrapper.eval()
    model_wrapper.generatorNet.eval()
    
    for i in idxList:
        show_results3(device, model_wrapper, testData, idx=i, size=size)

def colorizeFromImage(device, modelPath: str, imagePath: str, size):
    model = GANColorization(device, generatorNet=CONFIG.GeneratorTypes.RESNET)

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
        print("[MAIN] Start training with [torchrun --nproc_per_node=<nGPU>]|python main.py train <args>")
    else:
        trainData = COCO_LAB(size=CONFIG.IMAGE_SIZE, split="train")
        testData = COCO_LAB(size=CONFIG.IMAGE_SIZE, split="val")

        if CONFIG.MODE == CONFIG.RunMode.TRAIN:
            print("[MAIN] In train mode!")
            model = GANTrainer(CONFIG.DEVICE, trainData, testData, batchSize = CONFIG.BATCH_SIZE, numEpochs = CONFIG.NUM_EPOCH, displayEvery=100, pretrainedGeneratorPath=CONFIG.PRETRAINED_GENERATOR_PATH, generatorType=CONFIG.GENERATOR_TYPE)
            if CONFIG.LOCAL_RANK == 0:
                saveModel(model.module, CONFIG.NUM_EPOCH) 
                show_results3(CONFIG.DEVICE, model.module, testData, idx = 1, size=CONFIG.IMAGE_SIZE)
        elif CONFIG.MODE == CONFIG.RunMode.PRETRAIN:
            print("[MAIN] In pretrain mode!")
            model = pretrainGenerator(CONFIG.DEVICE, CONFIG.GENERATOR_TYPE, trainData, batchSize=CONFIG.BATCH_SIZE, numEpochs=CONFIG.NUM_EPOCH)
            if CONFIG.LOCAL_RANK == 0:
                saveModel(model.module, CONFIG.NUM_EPOCH)
        elif CONFIG.MODE == CONFIG.RunMode.INFER:
            print("[MAIN] In inferance mode!")
            if CONFIG.LOCAL_RANK == 0:
                displaySelectedResults3(CONFIG.DEVICE, CONFIG.MODEL_PATH, testData, CONFIG.IMG_TO_DISPLAY, size=CONFIG.IMAGE_SIZE)

            # elif mode == 4:
            #     if CONFIG.LOCAL_RANK == 0:
            #         imagePath = "examples/example1_bw.jpeg"
            #         colorizeFromImage(CONFIG.DEVICE, modelPath, imagePath, size=imSize)

            # elif mode == 8:
            #     if CONFIG.LOCAL_RANK == 0:
            #         displayIndustrialModelResults(CONFIG.DEVICE, testData, fullRangeToDisplay, imSize)

        if CONFIG.USE_DDP:
            tdist.destroy_process_group()
    
