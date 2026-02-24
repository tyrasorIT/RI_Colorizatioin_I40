import torch
import csv
import os

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    lossDiscriminatorFake = AverageMeter()
    lossDiscriminatorReal = AverageMeter()
    lossDiscriminator = AverageMeter()
    lossGeneratorGAN = AverageMeter()
    lossGeneratorL1 = AverageMeter()
    lossGenerator = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()

    return {'lossDiscriminatorFake': lossDiscriminatorFake,
            'lossDiscriminatorReal': lossDiscriminatorReal,
            'lossDiscriminator': lossDiscriminator,
            'lossGeneratorGAN': lossGeneratorGAN,
            'lossGeneratorL1': lossGeneratorL1,
            'lossGenerator': lossGenerator,
            'psnr': psnr,
            'ssim': ssim
            }


def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

def reduce_meter_dict(meterDict, device):
    for meter in meterDict.values():
        sumTensor = torch.tensor(meter.sum, device=device)
        countTensor = torch.tensor(meter.count, device=device)

        torch.distributed.all_reduce(sumTensor)
        torch.distributed.all_reduce(countTensor)

        meter.sum = sumTensor.item()
        meter.count = countTensor.item()
        meter.avg = meter.sum / meter.count if meter.count > 0 else 0.0

class CSVLogger:
    def __init__(self, filePath, fieldNames):
        self.filePath = filePath
        self.fieldNames = fieldNames

        fileExists = os.path.isfile(filePath)
        self.file = open(filePath, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldNames)

        if not fileExists:
            self.writer.writeheader()

    def log(self, rowDict):
        self.writer.writerow(rowDict)
        self.file.flush()

    def close(self):
        self.file.close()