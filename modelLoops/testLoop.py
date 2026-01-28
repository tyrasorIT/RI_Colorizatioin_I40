import torch

def test_loop(model, loader, lossFn, device):
    model.eval()
    totalLoss = 0

    with torch.no_grad():
        for l, ab in loader:
            x = l.to(device)
            y = ab.to(device)

            predAB = model(x)
            predAB = torch.clamp(predAB, -1, 1)

            loss = lossFn(predAB, y, x)
            totalLoss += loss.item()

    print("Total loss (test): ", totalLoss/len(loader))

    return totalLoss