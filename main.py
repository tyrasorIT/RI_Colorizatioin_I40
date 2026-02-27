import torch
import torch.distributed as tdist
from utils.dataset import COCO_LAB
from pretrainers.pretrainGenerator import pretrainGenerator
from trainers.GANTrainer import GANTrainer
from utils.display import displaySelectedResults, displayIndustrialModelResults
from utils.model import saveModel
from config import CONFIG

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
            model = GANTrainer(CONFIG.DEVICE, trainData, testData, generatorType=CONFIG.GENERATOR_TYPE, batchSize = CONFIG.BATCH_SIZE, numEpochs = CONFIG.NUM_EPOCH, displayEvery=100)
            if CONFIG.LOCAL_RANK == 0:
                saveModel(model.module) 
        elif CONFIG.MODE == CONFIG.RunMode.PRETRAIN:
            print("[MAIN] In pretrain mode!")
            model = pretrainGenerator(CONFIG.DEVICE, CONFIG.GENERATOR_TYPE, trainData, batchSize=CONFIG.BATCH_SIZE, numEpochs=CONFIG.NUM_EPOCH)
            if CONFIG.LOCAL_RANK == 0:
                saveModel(model.module)
        elif CONFIG.MODE == CONFIG.RunMode.INFER:
            print("[MAIN] In inferance mode!")

            if CONFIG.INFER_MODE == CONFIG.InferMode.TRAINED:
                print("[MAIN] Trained model inferance mode!")
                if CONFIG.LOCAL_RANK == 0:
                    displaySelectedResults(CONFIG.DEVICE, CONFIG.MODEL_PATH, testData, CONFIG.IMG_TO_DISPLAY, size=CONFIG.IMAGE_SIZE)
            elif CONFIG.INFER_MODE == CONFIG.InferMode.INDUSTRIAL:
                print("[MAIN] Industrial model inferance mode!")
                if CONFIG.LOCAL_RANK == 0:
                    displayIndustrialModelResults(CONFIG.DEVICE, testData, CONFIG.IMG_TO_DISPLAY, CONFIG.IMAGE_SIZE)

        if CONFIG.USE_DDP:
            tdist.destroy_process_group()
    
