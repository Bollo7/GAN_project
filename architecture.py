import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# DCGAN and WGAN implementations based on https://github.com/Natsu6767/DCGAN-PyTorch/blob/master/dcgan.py

def weights_init(w):
    """
    Randomly initializes weights for convolutional and bactch normalization layers
    """
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, noise_z, ngf, n_channels, kernel_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(

            nn.ConvTranspose2d(noise_z, ngf * 8, kernel_size=kernel_size, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 2x2
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=kernel_size, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 4x4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 16x16
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 32x32
            nn.ConvTranspose2d(ngf, n_channels, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.Tanh()
            # # 64x64
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ndf, nc, kernel_size):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 64x64
            nn.Conv2d(nc, ndf, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32
            nn.Conv2d(ndf, ndf * 2, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4
            nn.Conv2d(ndf * 8, 1, kernel_size=kernel_size, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output

# Modified DCGAN architecture to be WGAN

class W_Generator(nn.Module):
    def __init__(self, noise_z, ngf, n_channels, kernel_size):
        super(W_Generator, self).__init__()
        self.main = nn.Sequential(

            nn.ConvTranspose2d(noise_z, ngf * 8, kernel_size=kernel_size, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 2x2
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=kernel_size, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 4x4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 16x16
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 32x32
            nn.ConvTranspose2d(ngf, n_channels, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.Tanh()
            # # 64x64
        )

    def forward(self, input):
        output = self.main(input)
        return output


class W_Discriminator(nn.Module):
    def __init__(self, ndf, nc, kernel_size):
        super(W_Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 64x64
            nn.Conv2d(nc, ndf, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32
            nn.Conv2d(ndf, ndf * 2, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4
            nn.Conv2d(ndf * 8, 1, kernel_size=kernel_size, stride=1, padding=0, bias=False)
        )

    def forward(self, input):
        output = self.main(input)
        output = output.mean(0)
        return output.view(1)
    
    def extract_features(self, input):
        input = self.main(input)
        return input.view(-1)
