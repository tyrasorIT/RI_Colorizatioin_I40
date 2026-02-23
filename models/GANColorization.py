from torch import nn
import torch
from utils.model import initModel
from .ResNetUNetColorization import ResNetUNetColorization
from .UNetColorization import UNetColorization
from .discriminators import PatchDiscriminator
from losses.customGANLoss import customGANLoss
from torch import optim

class GANColorization(nn.Module):
    def __init__(self, device, generatorNet=None, lrGenerator=2e-4, lrDiscriminator=2e-4, beta1=0.5, beta2=0.999, lambdaL1=100.):
        super().__init__()

        self.device = device
        self.lambdaL1 = lambdaL1

        if generatorNet is None:
            print("GAN using Unet.")
            self.generatorNet = initModel(UNetColorization(in_ch= 1, out_ch= 2, base_ch=64), self.device)
        elif generatorNet == "resnet":
            print("GAN using ResNet.")
            self.generatorNet = ResNetUNetColorization(out_ch=2, pretrained=True).to(self.device)
        else:
            print("GAN using sent generator.")
            self.generatorNet = generatorNet.to(self.device)
        self.discriminatorNet = initModel(PatchDiscriminator(inCh=3, numFilters=64, nDown=3), self.device)
        self.GANCriterion = customGANLoss(ganMode= "vanilla").to(self.device)
        self.L1Criterion = nn.L1Loss()
        self.generatorOptimizer = optim.Adam(self.generatorNet.parameters(), lr=lrGenerator, betas=(beta1, beta2))
        self.discriminatorOptimizer = optim.Adam(self.discriminatorNet.parameters(), lr=lrDiscriminator, betas=(beta1, beta2)) # Wa lrGenerator

    def setRequiresGrad(self, model, requiresGrad=True):
        for p in model.parameters():
            p.requires_grad = requiresGrad

    def setupInput(self, data):
        L, ab = data  # or data[0], data[1]
        self.L = L.to(self.device)
        self.ab = ab.to(self.device)

    def forward(self):
        self.fakeColor = self.generatorNet(self.L)

    def backwardDiscriminator(self):
        fakeImage = torch.cat([self.L, self.fakeColor], dim=1)
        fakePreds = self.discriminatorNet(fakeImage.detach())
        self.lossDiscriminatorFake = self.GANCriterion(fakePreds, False)
        realImage = torch.cat([self.L, self.ab], dim=1)
        realPreds = self.discriminatorNet(realImage)
        self.lossDiscriminatorReal = self.GANCriterion(realPreds, True)
        self.lossDiscriminator = (self.lossDiscriminatorFake + self.lossDiscriminatorReal) * 0.5
        self.lossDiscriminator.backward()

    def backwardGenerator(self):
        fakeImage = torch.cat([self.L, self.fakeColor], dim=1)
        fakePreds = self.discriminatorNet(fakeImage)
        self.lossGeneratorGAN = self.GANCriterion(fakePreds, True)
        self.lossGeneratorL1 = self.L1Criterion(self.fakeColor, self.ab) * self.lambdaL1
        self.lossGenerator = self.lossGeneratorGAN + self.lossGeneratorL1
        self.lossGenerator.backward()

    def optimize(self):
        self.forward()
        self.discriminatorNet.train()
        self.setRequiresGrad(self.discriminatorNet, True)
        self.discriminatorOptimizer.zero_grad()
        self.backwardDiscriminator()
        self.discriminatorOptimizer.step()

        self.generatorNet.train()
        self.setRequiresGrad(self.discriminatorNet, False)
        self.generatorOptimizer.zero_grad()
        self.backwardGenerator()
        self.generatorOptimizer.step()