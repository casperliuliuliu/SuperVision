import torch.nn as nn
from torchvision import models
from Models.simple_CNN import SimpleCNN


def get_model(model_name, num_class_counts):
    # model = models.resnet18()
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_class_counts)
    model = SimpleCNN()
    num_ftrs = model.fc1.in_features
    model.fc1 = nn.Linear(num_ftrs, num_class_counts)

    return model