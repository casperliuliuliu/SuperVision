# Package
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
from torch.utils.data import ConcatDataset


def count_parameters(model):
    total_num = 0

    for parameter in model.parameters():
        if parameter.requires_grad:
            total_num += parameter.numel() 
    return total_num

def get_dataloaders(data_dir, data_transforms, train_ratio, val_ratio, batch_size):
    if datasets_is_split(data_dir):
        train_path = os.path.join(data_dir, "train")
        val_path = os.path.join(data_dir, "val")
        test_path = os.path.join(data_dir, "test")

    else:
        same_dir = data_dir

        train_path = same_dir
        val_path = same_dir
        test_path = same_dir

    train_dataset = datasets.ImageFolder(train_path, transform = data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_path, transform = data_transforms['val'])
    test_dataset = datasets.ImageFolder(test_path, transform = data_transforms['test'])

    for i, class_name in enumerate(train_dataset.classes):
        print(f"Class label {i}: {class_name}")

    if not datasets_is_split(data_dir):
        num_train = len(test_dataset)
        indices = list(range(num_train))
        print("--------- INDEX checking ---------")
        print(f"Original: {indices[:5]}")
        random.shuffle(indices)
        print(f"Shuffled: {indices[:5]}")
        print("--------- INDEX shuffled ---------\n")

        split_train = int(np.floor(train_ratio * num_train))
        split_val = split_train + int(np.floor(val_ratio * (num_train-split_train)))
        train_idx, val_idx, test_idx = indices[0:split_train], indices[split_train:split_val], indices[split_val:]
        train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(val_dataset, val_idx)
        test_dataset = Subset(test_dataset, test_idx)
        
        for ii in range(len(data_transforms.keys())-3):
            aug_dataset = datasets.ImageFolder(data_dir, transform = data_transforms[f'aug{ii}'])
            aug_sub = Subset(aug_dataset, train_idx)
            train_dataset = ConcatDataset([train_dataset, aug_sub])
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
    # check dataset
    print(f"Total number of samples: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)} datapoints")
    print(f"Number of train samples: {len(train_loader)} batches/ {len(train_loader.dataset)} datapoints")
    print(f"Number of val samples: {len(val_loader)} batches/ {len(val_loader.dataset)} datapoints")
    print(f"Number of test samples: {len(test_loader)} batches/ {len(test_loader.dataset)} datapoints")
    print(f"Data Transform: {data_transforms.keys()}\n")
    
    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    return dataloaders
import os

def datasets_is_split(path):
    folders_to_check = ['train', 'val', 'test']

    for folder in folders_to_check:
        folder_path = os.path.join(path, folder)
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            return False

    return True



if __name__ == "__main__":
    data_dir = "D:/Casper/Data/Animals-10/raw-img"
    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
            ]),
            'val':transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
            ]),
            'test':transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
            ]),
        }
    train_ratio = 0.6
    val_ratio = 0.5
    batch_size = 100

    torch.manual_seed(645)
    random.seed(645)
    dataloader = get_dataloaders(data_dir, data_transforms, train_ratio, val_ratio, batch_size)
    print(dataloader)