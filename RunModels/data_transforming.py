from torchvision import transforms

def get_data_transform(data_transform_name):
    data_transform = {
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
            }
    if data_transform_name == "basic":
        pass
    elif data_transform_name == "basic_aug":
        data_transform['aug0'] = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
    return data_transform