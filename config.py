import os
import torch

class Config:
    def __init__(self):
        self.use_ddp = False
        self.local_rank = 0 #Because of tqdm
        self.device = self._getDevice()

    def _getDevice(self):
        if 'LOCAL_RANK' in os.environ:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
            self.use_ddp = True
            if self.local_rank == 0:
                print("Using multiGPU setup!")
        elif torch.cuda.is_available():
            device = torch.device('cuda:0')
            print("Using single GPU setup!")
        else:
            device = torch.device('cpu')
            print("Warning: Using CPU!")

        return device
    
    @property
    def DEVICE(self):
        return self.device
    
    @property
    def LOCAL_RANK(self):
        return self.local_rank
    
    @property
    def USE_DDP(self):
        return self.use_ddp

CONFIG = Config()