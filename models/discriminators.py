import torch.nn as nn
import torch

class PatchDiscriminator(nn.Module):
    def __init__(self, inCh, numFilters=64, nDown=3):
        super().__init__()
        model = [self.getLayers(inCh, numFilters, normalization=False)]
        model += [
            self.getLayers(numFilters * 2 ** i, numFilters * 2 ** (i + 1), stride=1 if i == (nDown - 1) else 2)
                for i in range(nDown)
        ]
        model += [self.getLayers(numFilters * 2 ** nDown, 1, stride=1, normalization=False, activation=False)]

        self.model = nn.Sequential(*model)


    def getLayers(self, inCh, outCh, kernelSize = 4, stride=2, padding=1, normalization=True, activation=True):
        layers = [nn.Conv2d(inCh, outCh, kernelSize, stride, padding, bias=not normalization)]
        if normalization:
            layers += [nn.BatchNorm2d(outCh)]
        if activation:
            layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)