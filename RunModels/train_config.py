def get_config():


    config = {
        'model_name' : 'sth',
        'criterion_name' : "sth",
        'optimizer_name' : "sth",
        'lr_scheduler_name' : "sth",

        'data_dir' : "D:/Casper/Data/Animals-10/raw-img",
        'train_ratio' : 0.6,
        'val_ratio' : 0.5,
        'random_seed' : 42,
        'batch_size' : 4,
        'learning_rate' : 0.01,
        'num_of_epoch' : 20,
        'random_seed' : 645,
        'num_per_class' : 10,
        'classes_list' : ['cane','cavallo','elefante','gallina'],

        'pretrain' : True,
        'pretrain_category' : None,
        'data_transform_name' : "sth",

        'other_info' : "To build training functions",
        'log_file_path': 'D:/Casper/Log/log_0108.txt',
    }


    return config