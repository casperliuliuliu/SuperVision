import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, (11,11), stride=4)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((3,3), stride=2)
        self.conv2 = nn.Conv2d(96, 256, (5,5), padding=2)
        self.conv3 = nn.Conv2d(256, 384, (3,3), padding=1)
        self.conv4 = nn.Conv2d(384, 384, (3,3), padding=1)
        self.conv5 = nn.Conv2d(384, 256, (3,3), padding=1)
        self.fc1 = nn.Linear(6400, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = torch.flatten(x, 0)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.fc3(x)
        
        return x
if __name__ == "__main__":
    img = torch.rand(size=(3, 224,224)) *256
    model = AlexNet()
    result = model(img)
    print(model(img).shape)