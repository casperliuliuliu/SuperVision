
import time
import copy
import torch
from tqdm import tqdm
from torchvision import transforms
from RunModels.cprint import pprint
from Models.get_model import get_model
from RunModels.data_transforming import get_data_transform
from RunModels.methods import get_criterion, get_optimizer, get_lr_scheduler
from RunModels.basics import get_dataset_sizes, get_class_count, count_parameters, format_number
from RunModels.data_loader import get_dataloaders
def train_model(model_things):
    class_count = get_class_count(model_things['data_dir'])
    num_class = len(class_count)
    
    model = get_model(model_things['model_name'], num_class)
    criterion = get_criterion(model_things['criterion_name'])
    optimizer = get_optimizer(model_things['optimizer_name'], model)
    lr_scheduler = get_lr_scheduler(model_things['lr_scheduler_name'], optimizer)
    parameters_num = count_parameters(model)
    pprint(f"Total number of parameter in model: {format_number(parameters_num)}")

    data_dir = model_things['data_dir']
    train_ratio = model_things['train_ratio']
    val_ratio = model_things['val_ratio']
    batch_size = model_things['batch_size']
    num_of_epoch = model_things['num_of_epoch']
    random_seed = model_things['random_seed']
    num_per_class = model_things['num_per_class']
    classes_list = model_things['classes_list']
 
    data_transforms = get_data_transform(model_things['data_transform_name'])
    dataloaders = get_dataloaders(data_dir, data_transforms, train_ratio, val_ratio, batch_size, random_seed,  num_per_class, classes_list)
    dataset_sizes = get_dataset_sizes(dataloaders)    
    model = model.cuda()
    start_time = time.time()
    
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
    time_elapsed = time.time() - start_time
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
                transforms.Resize(224, 224),
                transforms.ToTensor(),
            ]),
            'val':transforms.Compose([
                transforms.Resize(224, 224),
                transforms.ToTensor(),
            ]),
            'test':transforms.Compose([
                transforms.Resize(224, 224),
                transforms.ToTensor(),
            ]),
        }

    model_things['num_class'] =  len(model_things['class_count'])
    train_model(model_things)