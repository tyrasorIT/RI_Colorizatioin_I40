from models.GANColorization import GANColorization
from config import CONFIG
from typing import List
from utils.model import loadModelCheckpoint
from utils.results import show_results3, show_results4, _colorizeFromImage
from external.colorization.colorizers import *
import os

def displaySelectedResults(device, modelPath: str, testData, idxList: List[int], size, generatorType=None):

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

    with open(os.path.join(CONFIG.RESULTS_DIR, "experiment_info.txt"), "a") as f:
            f.write(f"Mode: {CONFIG.MODE.value}\n")
            f.write(f"SubMode: {CONFIG.INFER_MODE.value}\n")
            f.write(f"Generator Type: {generatorType.value}\n")
            f.write(f"Image Size: {CONFIG.IMAGE_SIZE}\n")
            f.write(f"Device: {torch.cuda.get_device_name(0)}\n")
            f.write(f"Torch Version: {torch.__version__}\n")
            f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(f"LocalModelPath: {CONFIG.MODEL_PATH}\n")

def displayIndustrialModelResults(device, testData, idxList: List[int], size):
    colorizer_eccv16 = eccv16(pretrained=True).to(device)
    colorizer_siggraph17 = siggraph17(pretrained=True).to(device)

    for i in idxList:
        show_results4(device, colorizer_eccv16, colorizer_siggraph17, testData, idx=i, size=size)

    with open(os.path.join(CONFIG.RESULTS_DIR, "experiment_info.txt"), "a") as f:
            f.write(f"Mode: {CONFIG.MODE.value}\n")
            f.write(f"SubMode: {CONFIG.INFER_MODE.value}\n")
            f.write(f"Image Size: {CONFIG.IMAGE_SIZE}\n")
            f.write(f"Device: {torch.cuda.get_device_name(0)}\n")
            f.write(f"Torch Version: {torch.__version__}\n")
            f.write(f"CUDA Version: {torch.version.cuda}\n")

def colorizeFromImage(device, modelPath: str, imagePath: str, size):
    model = GANColorization(device, generatorNet=CONFIG.GeneratorTypes.RESNET)

    loadModelCheckpoint(device, modelPath, model)

    _colorizeFromImage(device, model, imagePath, size)