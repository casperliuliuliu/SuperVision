import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler



def get_optimizer(optimizer_name, model, learning_rate=0.01, momentum=0):
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        print(f"[!!!] No such optimizer as {optimizer_name}")
    return optimizer

def get_criterion(criterion_name):
    if criterion_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    else:
        print(f"[!!!] No such criterion as {criterion_name}")
    return criterion

def get_lr_scheduler(scheduler_name, optimizer, step_size=50, gamma=0.9):
    if scheduler_name == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        print(f"[!!!] No such scheduler as {scheduler_name}")
    return scheduler