import torch.nn as nn
from torchvision import models
from Models.simple_CNN import SimpleCNN
from Models.AlexNet import AlexNet
def modify_last_layer(model, new_output_size):
    layers = list(model.children())
    for layer in reversed(layers):
        if hasattr(layer, 'out_features'):
            last_layer_input_features = layer.in_features
            break
        print("[!!!]: The layer modified was not last one, please check it.")
    
    for name, layer in reversed(list(model.named_children())):
        if hasattr(layer, 'out_features'):
            setattr(model, name, nn.Linear(last_layer_input_features, new_output_size))
            break
    return model

def get_model_structure(model_name, pretrain):
    if model_name == "resnet18":
        model = models.resnet18(weights=pretrain)
    elif model_name == "SimpleCNN":
        model = SimpleCNN()
    elif model_name == "alexnet":
        model = AlexNet()

    return model

def get_model(model_name, num_class_counts, pretrain=None):
    model = get_model_structure(model_name, pretrain)
    if model_name == "resnet18":
        model = modify_last_layer(model, num_class_counts)

    return model
## what if i use my own pretrain weight? haven't solved
# test pretrain weight(before or after.)
# can i load model simply by pretrain weight? (so i don't have to change to model structure when load the weight)


if __name__ == "__main__":
    get_model('resnet18',100)

