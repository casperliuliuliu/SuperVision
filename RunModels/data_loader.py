import os
import numpy as np
# from cprint import pprint
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
    print(class_counts)


def keep_first_n_per_class(dataset, num_class_count=10):
    class_count = {}
    selected_indices = []

    for index, (_, label) in enumerate(dataset.samples):
        if class_count.get(label, 0) < num_class_count:
            selected_indices.append(index)
            class_count[label] = class_count.get(label, 0) + 1
    print(selected_indices)
    # print(class_counts)
    # get_num_class_count(dataset)

    # class_count = {}
    # index_list = []

    # for _, class_label in dataset.samples:
    #     if class_label not in class_count:
    #         class_count[class_label] = 0

    #     if class_count[class_label] < num_class_count:
    #         index_list.append(class_label)
    #         class_count[class_label] += 1

    # # indices_to_keep = [num_class_count - ii -1 for ii in range(num_class_count*2)]

    dataset.samples = [dataset.samples[i] for i in selected_indices]
    # # merged_dataset.targets = [merged_dataset.targets[i] for i in indices_to_keep]

    # # print(filtered_dataset)
    class_counts = {}
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    print(class_counts)
    # from collections import Counter
    # print(Counter(dataset.targets))
    # return filtered_dataset
    pass


def get_datasets(data_dir, data_transforms, train_ratio, val_ratio, random_seed, num_class_count):
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
    print(len(train_dataset))
    train_dataset = keep_first_n_per_class(train_dataset, num_class_count)
    print(len(train_dataset))
    val_dataset = datasets.ImageFolder(val_path, transform = data_transforms['val'])
    test_dataset = datasets.ImageFolder(test_path, transform = data_transforms['test'])

    # for i, class_name in enumerate(train_dataset.classes):
    #     pprint(f"Class label {i}: {class_name}")

    # if not datasets_is_split(data_dir):
    #     num_train = len(test_dataset)
    #     indices = list(range(num_train))
    #     # pprint("--------- INDEX checking ---------")
    #     # pprint(f"Original: {indices[:5]}")
    #     # random.seed(random_seed)
    #     # random.shuffle(indices)
    #     # pprint(f"Shuffled: {indices[:5]}")
    #     # pprint("--------- INDEX shuffled ---------\n")

    #     split_train = int(np.floor(train_ratio * num_train))
    #     split_val = split_train + int(np.floor(val_ratio * (num_train-split_train)))
    #     train_idx, val_idx, test_idx = indices[0:split_train], indices[split_train:split_val], indices[split_val:]
    #     train_dataset = Subset(train_dataset, train_idx)
    #     val_dataset = Subset(val_dataset, val_idx)
    #     test_dataset = Subset(test_dataset, test_idx)
        
    # for ii in range(len(data_transforms.keys())-3):
    #     aug_dataset = datasets.ImageFolder(train_path, transform = data_transforms[f'aug{ii}'])
    #     aug_sub = Subset(aug_dataset, train_idx)
    #     train_dataset = ConcatDataset([train_dataset, aug_sub])

    # return {
    #     'train' : train_dataset,
    #     'val' : val_dataset,
    #     'test' : test_dataset,
    # }

def get_dataloaders(data_dir, data_transforms, train_ratio, val_ratio, batch_size, random_seed, max_number_of_data=None, classes_list=None):
    
    datasets = get_datasets(data_dir, data_transforms, train_ratio, val_ratio, random_seed, max_number_of_data)
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=True)
        
    # pprint(f"Total number of samples: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)} datapoints")
    # pprint(f"Number of train samples: {len(train_loader)} batches/ {len(train_loader.dataset)} datapoints")
    # pprint(f"Number of val samples: {len(val_loader)} batches/ {len(val_loader.dataset)} datapoints")
    # pprint(f"Number of test samples: {len(test_loader)} batches/ {len(test_loader.dataset)} datapoints")
    # pprint(f"Data Transform: {data_transforms.keys()}\n")
    
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
    get_datasets(data_dir, data_transforms, 0.6, 0.5, 645, 20)