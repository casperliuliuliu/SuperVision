from torchvision.datasets.vision import VisionDataset
from torchvision import datasets


def reverse_dict(input_dict):
    return {value: key for key, value in input_dict.items()}

class FilteredDataset(VisionDataset):
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    def __init__(self, original_dataset, class_list):
        if isinstance(class_list[0], int):
            idx_to_class = reverse_dict(original_dataset.class_to_idx)
            class_idx_list = class_list.copy()
            class_list = []
            for ii in class_idx_list:
                class_list.append(idx_to_class[ii])

        classes = class_list
        self.classes = class_list

        class_to_idx = {}
        for ii, class_name in enumerate(classes):
            class_to_idx[class_name] = ii
        self.class_to_idx = class_to_idx


        self.loader = original_dataset.loader
        self.extensions = self.IMG_EXTENSIONS
        self.transform = original_dataset.transform
        self.target_transform = original_dataset.target_transform

        samples = datasets.folder.make_dataset(original_dataset.root, class_to_idx, self.IMG_EXTENSIONS)
        self.samples = samples
        self.targets = [sample[1] for sample in samples]
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)