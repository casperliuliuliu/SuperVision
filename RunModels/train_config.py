def get_config():


    config = {
        # 'model_name' : 'resnet18',
        'model_name' : 'alexnet',
        'criterion_name' : "CrossEntropyLoss",
        'optimizer_name' : "SGD",
        'lr_scheduler_name' : "StepLR",

        'data_dir' : "D:/Casper/Data/Animals-10/raw-img",
        'train_ratio' : 0.6,
        'val_ratio' : 0.5,
        'random_seed' : 42,
        'batch_size' : 1,
        'learning_rate' : 0.01,
        'num_of_epoch' : 20,
        'random_seed' : 645,
        'num_per_class' : 100,
        'classes_list' : [],
        'show_confus' : False,

        'pretrain' : True,
        'pretrain_category' : None,
        'data_transform_name' : "basic_aug",

        'other_info' : "Train AlexNet.",
        'log_file_path': 'D:/Casper/Log/log_0111.txt',
    }


    return config