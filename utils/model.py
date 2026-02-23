import torch
from torch import nn
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

def saveModel(savePath, model: torch.nn.Module, numEpochs):
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": numEpochs,
    }, savePath)

    print("Saved model to", savePath)

def loadModelCheckpoint(device, modelPath: str, model: torch.nn.Module):
    
    checkpoint = torch.load(modelPath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

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
    print(f"model initialized with {init}")
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