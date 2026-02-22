import torch

def getTorchInfo():
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())