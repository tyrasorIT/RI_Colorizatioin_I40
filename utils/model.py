import torch
from torch import nn
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
from config import CONFIG
from typing import Optional

def saveModel(model: torch.nn.Module, inferOnly=True):

    checkpoint = {"model_state_dict": model.state_dict()}

    if not inferOnly:
        if hasattr(model, "generatorNet"):
            checkpoint["generator_state_dict"] = model.generatorNet.state_dict()
        if hasattr(model, "discriminatorNet"):
            checkpoint["discriminator_state_dict"] = model.discriminatorNet.state_dict()
        if hasattr(model, "generatorOptimizer"):
            checkpoint["generator_optimizer_state_dict"] = model.generatorOptimizer.state_dict()
        if hasattr(model, "discriminatorOptimizer"):
            checkpoint["discriminator_optimizer_state_dict"] = model.discriminatorOptimizer.state_dict()

    checkpoint["epoch"] = CONFIG.NUM_EPOCH
    checkpoint["generator_type"] = CONFIG.GENERATOR_TYPE.value
    checkpoint["image_size"] = CONFIG.IMAGE_SIZE
    checkpoint["batch_size"] = CONFIG.BATCH_SIZE

    torch.save(checkpoint, CONFIG.MODEL_PATH)

    print("[MODEL SAVER] Saved model to", CONFIG.MODEL_PATH)

def loadModelCheckpoint(device, modelPath: str, model: Optional[torch.nn.Module], metadataOnly=False):
    
    checkpoint = torch.load(modelPath, map_location=device)
    if not metadataOnly:
        model.load_state_dict(checkpoint["model_state_dict"])

    #This is model metadata
    try:
        metadata = {
            "epoch": checkpoint["epoch"],
            "generator_type": checkpoint["generator_type"],
            "image_size": checkpoint["image_size"],
            "batch_size": checkpoint["batch_size"]
        }
    except Exception as e:
        print("[MODEL LOADER] Metadata missing!")
        metadata = {}
    
    return metadata

def initWeights(net, init='norm', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
    print(f"[MODEL WEIGHTS] model initialized with {init}")
    return net

def initModel(model, device):
    model = model.to(device)
    model = initWeights(model)
    return model

def buildResNetFastAi(device, n_input=1, n_output=2, size=256):
    model = resnet18(pretrained=True)
    body = create_body(model, pretrained=True, n_in=n_input, cut=-2)
    generatorNet = DynamicUnet(body, n_output, (size, size)).to(device)
    return generatorNet

def countParameters(model: torch.nn.Module, trainable_only=True):
    model_to_count = model.module if hasattr(model, "module") else model

    if trainable_only:
        return sum(p.numel() for p in model_to_count.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model_to_count.parameters())