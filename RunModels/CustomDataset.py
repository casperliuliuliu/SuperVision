class CustomFilteredDataset(Dataset):
    def __init__(self, original_dataset, class_list):
        self.original_dataset = original_dataset
        self.class_list = class_list

        # Filter indices based on class_list
        self.filtered_indices = [idx for idx, (_, label) in enumerate(original_dataset.samples) if label in class_list]

        # Update classes and class_to_idx to reflect only the classes in class_list
        classes = [original_dataset.classes[i] for i in class_list]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        print(original_dataset.root)
        samples = datasets.folder.make_dataset(original_dataset.root, class_to_idx, is_valid_file=original_dataset.folder.is_valid_file)
    
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        return self.original_dataset[self.filtered_indices[index]]

    def __len__(self):
        return len(self.filtered_indices)