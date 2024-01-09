def get_config():


    config = {
        'model_name' : 'resnet18',
        'criterion_name' : "CrossEntropyLoss",
        'optimizer_name' : "SGD",
        'lr_scheduler_name' : "StepLR",

        'data_dir' : "D:/Casper/Data/Animals-10/raw-img",
        'train_ratio' : 0.6,
        'val_ratio' : 0.5,
        'random_seed' : 42,
        'batch_size' : 4,
        'learning_rate' : 0.01,
        'num_of_epoch' : 5,
        'random_seed' : 645,
        'num_per_class' : -1,
        'classes_list' : [],

        'pretrain' : True,
        'pretrain_category' : None,
        'data_transform_name' : "basic_aug",

        'other_info' : "To build training functions",
        'log_file_path': 'D:/Casper/Log/log_0109.txt',
    }


    return config