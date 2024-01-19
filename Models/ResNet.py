import torch
from torch import nn


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (7,7), 2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3,3), 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3,3)),
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (3,3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (3,3)),
        )
        self.conv4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (3,3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (3,3)),
        )
        self.conv5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, (3,3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, (3,3)),
        )
        self.avg = nn.AvgPool2d(2,2)
        self.fc = nn.Linear(12,1000)
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        x = self.conv1(x)

        # last_x = x.clone()
        # x = self.Conv2d(64, 64, (3,3))
        # x = nn.ReLU(inplace=True)
        # x = self.Conv2d(64, 64, (3,3))
        # x = nn.ReLU(inplace=True)



        print(x.shape)
        return x

if __name__ == "__main__":
    img = torch.rand(size=(3, 224,224)) *256
    model = ResNet18()
    result = model(img)
    print(model(img).shape)