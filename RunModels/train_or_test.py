
from PPRINT import pprint
from basics import get_dataloaders, get_dataset_sizes, get_class_count
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets
import time
import copy
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from datetime import datetime
import random
from torchvision import datasets, transforms
from torchvision import models
from methods import get_model,get_criterion,get_optimizer,get_lr_scheduler
def train_model(model_things):
    num_class = model_things['num_class']
    model = get_model(model_things['model_name'], num_class)
    num_of_epoch = model_things['num_of_epoch']
    data_dir = model_things['data_dir']
    train_ratio = model_things['train_ratio']
    val_ratio = model_things['val_ratio']
    batch_size = model_things['batch_size']
    data_transforms = model_things['data_transforms']
    criterion = get_criterion(model_things['criterion_name'])
    optimizer = get_optimizer(model_things['optimizer_name'], model)
    lr_scheduler = get_lr_scheduler(model_things['lr_scheduler_name'], optimizer)

    dataloaders = get_dataloaders(data_dir, data_transforms, train_ratio, val_ratio, batch_size)
    dataset_sizes = get_dataset_sizes(dataloaders)
    
    model = model.cuda()
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_of_epoch):
        pprint('Epoch [%d/%d]'% (epoch+1, num_of_epoch), show_time=True)
        pprint('-' * 10)
        pprint("Learning rate:{}".format(optimizer.param_groups[0]['lr']))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            
            confus = torch.zeros(num_class, num_class,dtype=int)            
            for inputs, labels in tqdm(dataloaders[phase]): # Iterate over data.
                inputs, labels = inputs.cuda(), labels.cuda()
                with torch.set_grad_enabled(phase == 'train'): # forward # track history if only in train
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # This is for printing the probability of each preds
                    # probabilities = F.softmax(outputs, dim=1)
                    # print(probabilities)
                    loss = criterion(outputs, labels)
                    if phase == 'train': # backward + optimize only if in training phase
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                for ii in range(len(preds)):# statistics
                    confus[ labels.data[ii] ][ preds[ii] ]+=1
                    
            if phase == 'train':
                lr_scheduler.step()
                # pass
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            pprint(confus)
            pprint('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) 
        print()
    time_elapsed = time.time() - since
    pprint('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    pprint('Best val Acc: {:.4f}'.format(
                best_acc))
    # log_message += '\n  Whole training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60)
    # log_message +='\n Best val Acc={:.4f}'.format(
    #             best_acc)
    
    # send_email(log_message, model_name)
    
    pprint("="*20)
    pprint()
    model.load_state_dict(best_model_wts) # load best model weights
    return model

if __name__ == "__main__":
    pprint('')
    pprint('',show_time=True)

    model_things = {
        'data_dir' : "D:/Casper/NSYSU/P2023/DATA/glomer_cg",
        'train_ratio' : 0.6,
        'val_ratio' : 0.5,
        'random_seed' : 42,
        'batch_size' : 20,
        'learning_rate' : 0.01,
        'num_of_epoch' : 1,
        'pretrain' : True,
        'pretrain_category' : None,
        'model_name' : 'sth',
        'other_info' : "To build training functions",

           }
    model_things['criterion_name'] = "sth"
    model_things['optimizer_name'] = "sth"
    model_things['lr_scheduler_name'] = "sth"

    model_things['data_transforms'] = {
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

    model_things['class_count'] = get_class_count(model_things['data_dir'])
    model_things['num_class'] =  len(model_things['class_count'])
    train_model(model_things)