# Package
import os
import random
import numpy as np
from RunModels.cprint import pprint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset

def count_parameters(model):
    total_num = 0

    for parameter in model.parameters():
        if parameter.requires_grad:
            total_num += parameter.numel() 
    return total_num

def produce_message(model_things):
    # Change to write information with for loops.
    log_message = ""

#     learning_rate = model_things['learning_rate']
#     num_of_epoch = model_things['num_of_epoch']
#     data_dir = model_things['data_dir']
#     lr_method = model_things['lr_method']
#     train_ratio = model_things['train_ratio']
#     val_ratio = model_things['val_ratio']
#     weight_store_path = model_things['weight_store_path']
#     pretrain = model_things['pretrain']
#     batch_size = model_things['batch_size']
#     model_name = model_things['model_name']
#     other_info = model_things['other_info']
#     dropout_prob = model_things['dropout_prob']
#     data_transforms_op = model_things['data_transforms_op']
    
#     log_message = f"""
# Base:
#     model: {model_name}
#     Dataset: {class_counts}
#     Dataset dir: {data_dir}

# Train:
#     epoch: {num_of_epoch}
#     pretrained: {pretrain}
#     batch size: {batch_size}
#     learning rate: {learning_rate}
#     lr method: {lr_method}
#     split ratio: {train_ratio}
#     val/test ratio: {val_ratio}
#     dropout rate: {dropout_prob}
#     transform Opt: {data_transforms_op}
    
# Other Information:
#     {other_info}
    
#     weight dir: {weight_store_path}
# """
#     pprint(log_message, show_time=True)
    return log_message

def format_number(num):
    if num >= 1e12:
        return f"{num/1e12:.0f}T"
    elif num >= 1e9:
        return f"{num/1e9:.0f}B"
    elif num >= 1e6:
        return f"{num/1e6:.0f}M"
    elif num >= 1e3:
        return f"{num/1e3:.0f}K"
    else:
        return str(num)



def get_class_count(data_dir):
    if datasets_is_split(data_dir):
        train_path = os.path.join(data_dir, "train")
    else:
        train_path = data_dir

    train_dataset = datasets.ImageFolder(train_path)
    return train_dataset.classes

def get_dataset_sizes(dataloaders):
    dataset_sizes = {
        'train': len(dataloaders['train'].dataset),
        'val': len(dataloaders['val'].dataset),
        'test': len(dataloaders['test'].dataset)
    }
    return dataset_sizes
def get_datasets(data_dir, data_transforms, train_ratio, val_ratio, random_seed):
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
        pprint(f"Class label {i}: {class_name}")

    if not datasets_is_split(data_dir):
        num_train = len(test_dataset)
        indices = list(range(num_train))
        pprint("--------- INDEX checking ---------")
        pprint(f"Original: {indices[:5]}")
        random.seed(random_seed)
        random.shuffle(indices)
        pprint(f"Shuffled: {indices[:5]}")
        pprint("--------- INDEX shuffled ---------\n")

        split_train = int(np.floor(train_ratio * num_train))
        split_val = split_train + int(np.floor(val_ratio * (num_train-split_train)))
        train_idx, val_idx, test_idx = indices[0:split_train], indices[split_train:split_val], indices[split_val:]
        train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(val_dataset, val_idx)
        test_dataset = Subset(test_dataset, test_idx)
        
    for ii in range(len(data_transforms.keys())-3):
        aug_dataset = datasets.ImageFolder(train_path, transform = data_transforms[f'aug{ii}'])
        aug_sub = Subset(aug_dataset, train_idx)
        train_dataset = ConcatDataset([train_dataset, aug_sub])

    return {
        'train' : train_dataset,
        'val' : val_dataset,
        'test' : test_dataset,
    }

def get_dataloaders(data_dir, data_transforms, train_ratio, val_ratio, batch_size, random_seed, max_number_of_data=None, classes_list=None):
    
    datasets = get_datasets(data_dir, data_transforms, train_ratio, val_ratio, random_seed)
    train_loader = DataLoader(datasets['train'], batch_size=batch_size)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size)
        
    pprint(f"Total number of samples: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)} datapoints")
    pprint(f"Number of train samples: {len(train_loader)} batches/ {len(train_loader.dataset)} datapoints")
    pprint(f"Number of val samples: {len(val_loader)} batches/ {len(val_loader.dataset)} datapoints")
    pprint(f"Number of test samples: {len(test_loader)} batches/ {len(test_loader.dataset)} datapoints")
    pprint(f"Data Transform: {data_transforms.keys()}\n")
    
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
    # data_dir = "D:/Casper/Data/Animals-10/raw-img"
    # data_transforms = {
    #         'train': transforms.Compose([
    #             transforms.Resize(224),
    #             transforms.ToTensor(),
    #         ]),
    #         'val':transforms.Compose([
    #             transforms.Resize(224),
    #             transforms.ToTensor(),
    #         ]),
    #         'test':transforms.Compose([
    #             transforms.Resize(224),
    #             transforms.ToTensor(),
    #         ]),
    #     }
    # train_ratio = 0.6
    # val_ratio = 0.5
    # batch_size = 100

    # pprint('', show_time=True)
    # dataloader = get_dataloaders(data_dir, data_transforms, train_ratio, val_ratio, batch_size)
    # pprint(dataloader)

    # Example usage:
    print(format_number(12000))   # Output: 12K
    print(format_number(365123456))  # Output: 365M
