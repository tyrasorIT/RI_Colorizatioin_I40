import  torch
from torch.cuda.amp import autocast

def train_loop(model, loader, lossFn, optimizer, device, scaler):
    model.train()
    totalLoss = 0

    for l, ab in loader:
        x = l.to(device, non_blocking=True)
        y = ab.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(dtype=torch.float16):
            predAB = model(x)
            predAB = torch.clamp(predAB, -1, 1)
            loss = lossFn(predAB, y, x)
            #loss.backward()

        #optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totalLoss += loss.item()

    print("Total loss (train): ", totalLoss/len(loader))