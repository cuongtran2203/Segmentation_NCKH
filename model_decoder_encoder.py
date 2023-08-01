import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
S=32 
F=7
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class MobileNetV2_Segmentation(nn.Module):
    def __init__(self, num_classes=None):
        super(MobileNetV2_Segmentation, self).__init__()
        self.encoder = mobilenet_v2(pretrained=True).features
        classifer=mobilenet_v2(pretrained=True).classifier
        conv2=nn.Conv2d(1280,1000,kernel_size=1)
        conv2.weight.data.copy_(classifer[1].weight.data.view(1000,1280,1,1))
        conv2.bias.data.copy_(classifer[1].bias.data)
        self.decoder = nn.Sequential(
            conv2,
            nn.BatchNorm2d(1000),
            nn.Conv2d(1000,960,kernel_size=1),
            nn.BatchNorm2d(960),
            nn.ReLU(),
            nn.Conv2d(960,num_classes,kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.LeakyReLU(),
            # fc6.weight.data.copy_(classifier[1].weight.data.view(1000, 1280, 1, 1))
            # fc6.bias.data.copy_(classifier[1].bias.data)
            nn.ConvTranspose2d(num_classes,num_classes,kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ConvTranspose2d(num_classes,num_classes,kernel_size=8,stride=8,bias=False),
            nn.BatchNorm2d(num_classes),
          
        )
    def forward(self, x):
        x=self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        return x
