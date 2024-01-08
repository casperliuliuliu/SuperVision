# Package
import os
from torchvision import datasets
from RunModels.data_loader import datasets_is_split
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
    # print(format_number(12000))   # Output: 12K
    # print(format_number(365123456))  # Output: 365M
    pass
