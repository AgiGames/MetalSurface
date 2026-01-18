import torch
from torch import nn

class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class MetalSurface(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()

        self.enc1 = CNNBlock(in_ch, base)
        self.enc2 = CNNBlock(base, base*2)
        self.enc3 = CNNBlock(base*2, base*4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = CNNBlock(base*4, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = CNNBlock(base*4, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = CNNBlock(base*2, base)

        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, 3, 480, 480)
        e1 = self.enc1(x) # (B, 32, 480, 480)
        p1 = self.pool(e1) # (B, 32, 240, 240)
        e2 = self.enc2(p1) # (B, 64, 240, 240)
        p2 = self.pool(e2) # (B, 64, 120, 120)
        e3 = self.enc3(p2) # (B, 128, 120, 120)
        
        b = self.bottleneck(e3) # (B, 128, 120, 120)
        
        u2 = self.up2(b) # (B, 64, 240 240)
        cat2 = torch.cat([u2, e2], dim=1) # (B, 128, 240, 240)
        d2 = self.dec2(cat2) # (B, 64, 240, 240)
        
        u1 = self.up1(d2) # (B, 32, 480, 480)
        cat1 = torch.cat([u1, e1], dim=1) # (B, 64, 480, 480)
        d1 = self.dec1(cat1) # (B, 32, 480, 480)
        
        return self.out(d1) # (B, 1, 480, 480)