import torch

def saveModel(savePath, model: torch.nn.Module, numEpochs):
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": numEpochs,
    }, savePath)

    print("Saved model to", savePath)