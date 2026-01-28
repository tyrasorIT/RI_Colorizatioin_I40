import torch
import torch.nn as nn

class ChromaLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predAB, targetAB):
        # Compute chroma
        predC = torch.sqrt(predAB[:,0]**2 + predAB[:,1]**2 + 1e-8)  # [N,H,W]
        targetC = torch.sqrt(targetAB[:,0]**2 + targetAB[:,1]**2 + 1e-8)
        
        loss = torch.abs(predC - targetC)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # no reduction