# Package
import numpy as np
from torchvision import datasets

from torch.utils.data import DataLoader, Subset
import random
from torch.utils.data import ConcatDataset


def count_parameters(model):
    total_num = 0
    
    for parameter in model.parameters():
        if parameter.requires_grad:
            total_num += parameter.numel() 
    return total_num

# def get_dataloaders(data_dir, data_transforms, train_ratio, val_ratio, batch_size):
#     # Create a single merged dataset
#     train_dataset = datasets.ImageFolder(data_dir, transform = data_transforms['train'])
#     val_dataset = datasets.ImageFolder(data_dir, transform = data_transforms['val'])
#     test_dataset = datasets.ImageFolder(data_dir, transform = data_transforms['test'])
#     for i, class_name in enumerate(train_dataset.classes):
#         print(f"Class label {i}: {class_name}")
#     # obtain training indices that will be used for validation
#     num_train = len(test_dataset)
#     indices = list(range(num_train))
#     print("--------- INDEX checking ---------")
#     print(f"Original: {indices[:5]}")
#     random.shuffle(indices)
#     print(f"Shuffled: {indices[:5]}")
#     print("--------- INDEX shuffled ---------\n")

#     split_train = int(np.floor(train_ratio * num_train))
#     split_val = split_train + int(np.floor(val_ratio * (num_train-split_train)))
#     train_idx, val_idx, test_idx = indices[0:split_train], indices[split_train:split_val], indices[split_val:]
#     merge_dataset = Subset(train_dataset, train_idx)
#     # file_names = [path for path, _ in train_dataset.imgs]
#     for ii in range(len(data_transforms.keys())-3):
#         # print(ii)
#         aug_dataset = datasets.ImageFolder(data_dir, transform = data_transforms[f'aug{ii}'])
#         aug_sub = Subset(aug_dataset, train_idx)
#         merge_dataset = ConcatDataset([merge_dataset,aug_sub])
    
#     train_loader = DataLoader(merge_dataset, batch_size=batch_size)
#     val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=batch_size)
#     test_loader = DataLoader(Subset(test_dataset, test_idx), batch_size=batch_size)
    
#     # check dataset
#     print(f"Total number of samples: {num_train} datapoints")
#     print(f"Number of train samples: {len(train_loader)} batches/ {len(train_loader.dataset)} datapoints")
#     print(f"Number of val samples: {len(val_loader)} batches/ {len(val_loader.dataset)} datapoints")
#     print(f"Number of test samples: {len(test_loader)} batches/ {len(test_loader.dataset)} datapoints")
#     print(f"Data Transform: {data_transforms.keys()}")
#     print(f"")
    
#     dataloaders = {
#         "train": train_loader,
#         "val": val_loader,
#         "test": test_loader,
#     }
#     return dataloaders

if __name__ == "__main__":
    # data_dir, data_transforms, train_ratio, val_ratio, batch_size
    # get_dataloaders()