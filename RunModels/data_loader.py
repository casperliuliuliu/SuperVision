import os
import numpy as np
import random
import torch
from RunModels.cprint import pprint
from RunModels.CustomDataset import FilteredDataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset

def is_dataset_sorted_by_class(dataset):
    last_label = None
    for _, label in dataset.samples:
        if last_label is not None and label < last_label:
            return False
        last_label = label
    return True

def get_num_class_count(dataset):
    class_counts = {}
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    pprint(class_counts)

def intersection_of_lists(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    intersection_list = sorted(list(intersection))
    return intersection_list

def filter_dataset_by_classes_list(original_dataset, classes_list=None):
    selected_indices = [index for index, (_, label) in enumerate(original_dataset.samples) if label in classes_list]
    return selected_indices

def filter_dataset_by_num_per_class(original_dataset, num_per_class=-1):
    if num_per_class < 0:
        all_indices = list(range(len(original_dataset)))
        return all_indices
    
    class_count = {}
    selected_indices = []
    for index, (_, label) in enumerate(original_dataset.samples):
        if class_count.get(label, 0) < num_per_class:
            selected_indices.append(index)
            class_count[label] = class_count.get(label, 0) + 1
    return selected_indices

def filter_dataset_by_classes_list_and_n_per_class(original_dataset, num_per_class, classes_list):
    num_classes_list = []
    for class_name in classes_list:
        num_classes_list.append(original_dataset.class_to_idx[class_name])
    num_per_class_indices = filter_dataset_by_num_per_class(original_dataset, num_per_class)
    by_classes_list_indices = filter_dataset_by_classes_list(original_dataset, num_classes_list)
    intersection_list = intersection_of_lists(num_per_class_indices, by_classes_list_indices)
    sub_dataset = Subset(original_dataset, intersection_list)
    return sub_dataset


def get_datasets(data_dir, data_transforms, train_ratio, val_ratio, random_seed, num_per_class=-1, classes_list=None):
    if datasets_is_split(data_dir):
        train_path = os.path.join(data_dir, "train")
        val_path = os.path.join(data_dir, "val")
        test_path = os.path.join(data_dir, "test")

    else:
        same_dir = data_dir

        train_path = same_dir
        val_path = same_dir
        test_path = same_dir
    ori_train_dataset = datasets.ImageFolder(train_path, transform = data_transforms['train'])
    ori_val_dataset = datasets.ImageFolder(val_path, transform = data_transforms['val'])
    ori_test_dataset = datasets.ImageFolder(test_path, transform = data_transforms['test'])
    train_dataset = ori_train_dataset
    if classes_list:
        train_dataset = FilteredDataset(ori_train_dataset, classes_list)
        val_dataset = FilteredDataset(ori_val_dataset, classes_list)
        test_dataset = FilteredDataset(ori_test_dataset, classes_list)
   
    train_selected_indices = filter_dataset_by_num_per_class(train_dataset, num_per_class)
    train_dataset = Subset(ori_train_dataset, train_selected_indices)

    train_idx = list(range(len(ori_train_dataset)))
    
    if not datasets_is_split(data_dir):
        num_train = len(train_dataset)
        pprint(f"Total Available Data: {num_train} datapoints (Train + Val + Test, No Aug)")
        indices = train_dataset.indices

        random.seed(random_seed)
        random.shuffle(indices)

        split_train = int(np.floor(train_ratio * num_train))
        split_val = split_train + int(np.floor(val_ratio * (num_train-split_train)))
        train_idx, val_idx, test_idx = indices[0:split_train], indices[split_train:split_val], indices[split_val:]

        train_dataset = Subset(ori_train_dataset, train_idx)
        val_dataset = Subset(ori_val_dataset, val_idx)
        test_dataset = Subset(ori_test_dataset, test_idx)

    for ii in range(len(data_transforms.keys())-3):
        aug_dataset = datasets.ImageFolder(train_path, transform = data_transforms[f'aug{ii}'])
        aug_sub = Subset(aug_dataset, train_idx)
        train_dataset = ConcatDataset([train_dataset, aug_sub])

    return {
        'train' : train_dataset,
        'val' : val_dataset,
        'test' : test_dataset,
    }

def show_sample(data_loader):
    for _, label in data_loader:
        print(label)

def get_dataloaders(data_dir, data_transforms, train_ratio, val_ratio, batch_size, random_seed, num_per_class=-1, classes_list=None):
    torch.manual_seed(random_seed)
    
    datasets = get_datasets(data_dir, data_transforms, train_ratio, val_ratio, random_seed, num_per_class, classes_list)
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size)
        
    pprint(f"Total number of samples: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)} datapoints")
    pprint(f"Number of train samples: {len(train_loader)} batches/ {len(train_loader.dataset)} datapoints")
    pprint(f"Number of val samples: {len(val_loader)} batches/ {len(val_loader.dataset)} datapoints")
    pprint(f"Number of test samples: {len(test_loader)} batches/ {len(test_loader.dataset)} datapoints")

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    return dataloaders

def datasets_is_split(path):
    folders_to_check = ['train', 'val', 'test']

    for folder in folders_to_check:
        folder_path = os.path.join(path, folder)
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            return False

    return True

if __name__ == "__main__":
    print('jj')
    data_dir = "D:/Casper/Data/Animals-10/raw-img"
    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            'val':transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            'test':transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            'aug0': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
        }
    # get_datasets(data_dir, data_transforms, 0.6, 0.5, 645)
    # data_loaders = get_dataloaders(data_dir, data_transforms, 0.6, 1, 4, 645, 10, [0, 1, 2, 4])
    data_loaders = get_dataloaders(data_dir, data_transforms, 0.6, 0.5, 4, 645, 10, ['cane','cavallo'])
    # data_loaders = get_dataloaders(data_dir, data_transforms, 0.6, 1, 4, 645, 10, ['cane','cavallo','elefante','gallina'])

