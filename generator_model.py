import torch    
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act= "relu", use_dropout= False):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2 , padding=1, padding_mode='reflect')
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
        
class Generator(nn.Module):
    def __init__(self, in_channels = 3, features = 64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        ) # 256 -> 128x128

        self.down1 = UNetBlock(features, features*2, down=True, act="leaky", use_dropout=False)   # 256 -> 128x128
        self.down2 = UNetBlock(features*2, features*4, down=True, act="leaky", use_dropout=False) # 128 -> 64x64
        self.down3 = UNetBlock(features*4, features*8, down=True, act="leaky", use_dropout=False) # 64 -> 32x32
        self.down4 = UNetBlock(features*8, features*8, down=True, act="leaky", use_dropout=False) # 32 -> 16x16
        self.down5 = UNetBlock(features*8, features*8, down=True, act="leaky", use_dropout=False) # 16 -> 8x8
        self.down6 = UNetBlock(features*8, features*8, down=True, act="leaky", use_dropout=False) # 8 -> 4x4
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
        ) #1x1

        self.up1 = UNetBlock(features*8, features*8, down=False, act="relu", use_dropout=True) # 1x1 -> 2x2
        self.up2 = UNetBlock(features*8*2, features*8, down=False, act="relu", use_dropout=True)
        self.up3 = UNetBlock(features*8*2, features*8, down=False, act="relu", use_dropout=True)
        self.up4 = UNetBlock(features*8*2, features*8, down=False, act="relu", use_dropout=True)
        self.up5 = UNetBlock(features*8*2, features*4, down=False, act="relu", use_dropout=True)
        self.up6 = UNetBlock(features*4*2, features*2, down=False, act="relu", use_dropout=True)
        self.up7 = UNetBlock(features*2*2, features, down=False, act="relu", use_dropout=True)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Downsampling
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1,d7],1))
        up3 = self.up3(torch.cat([up2,d6],1))
        up4 = self.up4(torch.cat([up3,d5],1))
        up5 = self.up5(torch.cat([up4,d4],1))
        up6 = self.up6(torch.cat([up5,d3],1))
        up7 = self.up7(torch.cat([up6,d2],1))
        return self.final_up(torch.cat([up7,d1], 1))

def test():
    x = torch.randn((1,3,256,256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)

if __name__ == '__main__':
    test()
