import torch
import torch.nn as nn

class CNNlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features= [64, 128, 256, 512]): #256 -> 30x30
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNlock(in_channels, feature, stride = 1 if feature == features[-1] else 2)
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'),
        )
        self.mode = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.mode(x)
        return x
    
def test():
    x = torch.randn(1,3,256, 256)
    y = torch.randn(1,3,256, 256)
    model = Discriminator()
    preds = model(x, y)
    print(preds.shape) # (1, 512, 30, 30)

if __name__ == '__main__':
    test()