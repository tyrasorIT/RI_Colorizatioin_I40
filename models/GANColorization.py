from torch import nn
import torch
from utils.model import initModel
from utils.model import buildResNetFastAi, loadModelCheckpoint
from .ResNetUNetColorization import ResNetUNetColorization
from .discriminators import PatchDiscriminator
from losses.customGANLoss import customGANLoss
from torch import optim
from config import Config # Just for generator types
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure
import torch.nn.functional as tfunc
from utils.results import lab_to_rgb

class GANColorization(nn.Module):
    def __init__(self, device, generatorType=None, pretrained=False, pretrainedPath=None, lrGenerator=2e-4, lrDiscriminator=2e-4, beta1=0.5, beta2=0.999, lambdaL1=100.):
        super().__init__()

        self.device = device
        self.lambdaL1 = lambdaL1

        if generatorType is None or generatorType == Config.GeneratorTypes.RESNET:
            print("[GAN Colorization] Using ResNet generator.")
            self.generatorNet = ResNetUNetColorization(out_ch=2, pretrained=True).to(self.device)
        elif generatorType == Config.GeneratorTypes.FASTAI:
            print("[GAN Colorization] Using Fastai generator.")
            self.generatorNet = buildResNetFastAi(device, n_input=1, n_output=2, size=128)
        else:
            raise RuntimeError(f"[GAN Colorization] No proper generator selected!")
        
        if pretrained:
            print("[GAN Colorization] Using pretrained generators.")
            metadata = loadModelCheckpoint(self.device, pretrainedPath, self.generatorNet)
            if metadata:
                if Config.GeneratorTypes(metadata["generator_type"]) != generatorType:
                    raise RuntimeError(f"[GAN Colorization] Pretrained generator and generator type mismatch!")

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

    def compute_metrics(self):
        fake_lab = torch.cat([self.L, self.fakeColor], dim=1)
        real_lab = torch.cat([self.L, self.ab], dim=1)

        fake_rgb = lab_to_rgb(fake_lab)
        real_rgb = lab_to_rgb(real_lab)

        fake_rgb = torch.clamp(fake_rgb, 0., 1.)
        real_rgb = torch.clamp(real_rgb, 0., 1.)

        self.psnr = peak_signal_noise_ratio(fake_rgb, real_rgb, data_range=1.0)
        self.ssim = structural_similarity_index_measure(fake_rgb, real_rgb, data_range=1.0)