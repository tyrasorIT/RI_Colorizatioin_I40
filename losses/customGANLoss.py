import torch.nn as nn
import torch

class customGANLoss(nn.Module):
    def __init__(self, ganMode = "vanilla", realLabel = 1.0, fakeLabel = 0.0):
        super().__init__()

        self.register_buffer('real_label', torch.tensor(realLabel))
        self.register_buffer('fake_label', torch.tensor(fakeLabel))
        if ganMode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif ganMode == "lsgan":
            self.loss = nn.MSELoss()

    def getLabels(self, predictions, isReal):
        if isReal:
            labels = self.real_label
        else:
            labels = self.fake_label
        
        return labels.expand_as(predictions)
    
    def __call__(self, predictions, isReal):
        labels = self.getLabels(predictions, isReal)
        loss = self.loss(predictions, labels)
        return loss