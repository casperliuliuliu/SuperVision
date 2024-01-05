
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, resolution=256):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(1 * resolution * resolution, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        
        return x

if __name__ == "__main__":
    resolution = 256
    image = torch.randint(256, size=[3, resolution,resolution]).float()
    model = SimpleCNN(resolution)
    result = model(image)

    # print(result.shape)
    # print("")
    # total_num = 0
    
    # print("Parameter:")
    # for parameter in model.parameters():
    #     print(parameter.shape)
        # total_num += sum(parameter.shape)
        # print(total_num)
    print(count_parameters(model))
