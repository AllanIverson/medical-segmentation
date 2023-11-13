import torch
# import torchvision.transforms
import torch.nn as nn
import torch.nn.functional as F
KernelSize = 3
Pool_stride = 2

class Double_conv(nn.Module):
    #conv->bn->relu->conv->bn->relu
    def __init__(self, in_ch, out_ch):
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=KernelSize, padding=1),
            nn.BatchNorm3d(out_ch),
            # nn.GroupNorm(32,out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=KernelSize, padding=1),
            nn.BatchNorm3d(out_ch),
            # nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=Pool_stride),
            Double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.upcv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv = Double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.upcv(x1)
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)  # (B,C,H,W)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self,num_classes):
        super(Unet, self).__init__()
        self.inc = Double_conv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc = OutConv(32, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero()