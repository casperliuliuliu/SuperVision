

def get_config():
    data = {
        'model_name' : 'sth',
        'criterion_name' : "sth",
        'optimizer_name' : "sth",
        'lr_scheduler_name' : "sth",

        'data_dir' : "D:/Casper/NSYSU/P2023/DATA/glomer_cg",
        'train_ratio' : 0.6,
        'val_ratio' : 0.5,
        'random_seed' : 42,
        'batch_size' : 20,
        'learning_rate' : 0.01,
        'num_of_epoch' : 1,
        'pretrain' : True,
        'pretrain_category' : None,
        'data_transform_name' : "sth",

        'other_info' : "To build training functions",
        'log_file_path': 'D:/Casper/Log/log_0106.txt',
    }
    return data