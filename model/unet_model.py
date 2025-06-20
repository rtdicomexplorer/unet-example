import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.maxpool(d1))
        d3 = self.down3(self.maxpool(d2))
        d4 = self.down4(self.maxpool(d3))

        bottleneck = self.bottleneck(self.maxpool(d4))

        up4 = self.up4(bottleneck)
        merge4 = torch.cat([up4, d4], dim=1)
        up4 = self.conv4(merge4)

        up3 = self.up3(up4)
        merge3 = torch.cat([up3, d3], dim=1)
        up3 = self.conv3(merge3)

        up2 = self.up2(up3)
        merge2 = torch.cat([up2, d2], dim=1)
        up2 = self.conv2(merge2)

        up1 = self.up1(up2)
        merge1 = torch.cat([up1, d1], dim=1)
        up1 = self.conv1(merge1)

        return self.final(up1)  # ohne Sigmoid
        return torch.sigmoid(self.final(up1)) # because of BCEWithLogitsLoss 
