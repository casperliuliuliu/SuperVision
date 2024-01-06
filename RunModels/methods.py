import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

def get_model(model_name, num_class_counts):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class_counts)
    return model

def get_optimizer(optimizer_name, model, learning_rate=0.01):
    return optim.SGD(model.parameters(), lr=learning_rate)

def get_criterion(criterion_name):
    return nn.CrossEntropyLoss()

def get_lr_scheduler(scheduler_name, optimizer, step_size=50, gamma=0.9):
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
