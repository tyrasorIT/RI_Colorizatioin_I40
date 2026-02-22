

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

    return {'lossDiscriminatorFake': lossDiscriminatorFake,
            'lossDiscriminatorReal': lossDiscriminatorReal,
            'lossDiscriminator': lossDiscriminator,
            'lossGeneratorGAN': lossGeneratorGAN,
            'lossGeneratorL1': lossGeneratorL1,
            'lossGenerator': lossGenerator}


def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")