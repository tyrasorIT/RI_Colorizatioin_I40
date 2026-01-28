import torch
print(torch.__version__)       # should end with +cu124
print(torch.version.cuda)      # should be 12.4
print(torch.cuda.is_available())  # True