import json
from yacs.config import CfgNode as CN

def _get_config(tissue_type, deconv, subtype, k_class, tissue_dir):
    config = CN()
    config.train = CN()
    config.train.lr = 0.0005
    config.train.epoch = 41
    config.train.val_iter = 10
    config.train.val_min_iter = 9

    config.data = CN()
    config.data.deconv = deconv
    config.data.save_model = f'./train_log/{tissue_type}/models'    # model saved
    config.data.ckpt = f'./train_log/{tissue_type}/ckpts'   # eval results saved
    config.data.tile_dir = f'./demo/data/{tissue_type}/tiles'   # path to tiles
    config.data.mask_dir = f'./demo/data/{tissue_type}/seg' # path to json segmentation file
    config.data.batch_size = 32
    config.data.tissue_dir = tissue_dir # tissue compartment directory
    config.data.max_cell_num = 256  # max cell number in a single tile for batch learning
    config.data.cell_dir = f'./demo/data/{tissue_type}/cell_proportion/type/{config.data.deconv}'   # path to cell proportion label

    config.model = CN()
    config.model.tissue_class = 3
    config.model.pretrained = True
    config.model.channels = 3
    config.model.k_class = k_class
    
    return config

def _get_cell_state_config(tissue_type, deconv, subtype, tissue_dir):
    config = CN()
    config.train = CN()
    config.train.lr = 0.0005
    config.train.epoch = 41
    config.train.val_iter = 5
    config.train.val_min_iter = 9
    config.train.state_epoch = 41

    config.data = CN()
    config.data.deconv = deconv
    config.data.save_model = f'./train_log/{tissue_type}/models'
    config.data.ckpt = f'./train_log/{tissue_type}/ckpts'
    config.data.tile_dir = f'./demo/data/{tissue_type}/tiles'
    config.data.mask_dir = f'./demo/data/{tissue_type}/seg'
    config.data.batch_size = 32
    config.data.tissue_dir = tissue_dir
    config.data.max_cell_num = 256
    with open(tissue_dir, 'r') as tissue_file:
        tc = json.load(tissue_file)

    config.data.cell_dir = f'./demo/data/{tissue_type}/cell_proportion/type/{config.data.deconv}'
    config.data.state_dir = f'./demo/data/{tissue_type}/cell_proportion/state/{config.data.deconv}'    # path to cell state directory

    config.model = CN()
    config.model.tissue_class = len(tc['list'])
    config.model.pretrained = True
    config.model.channels = 3
    config.model.k_class = len(tc['dict'])

    k_state = 0
    for key, value in tc['state'].items():
        k_state += int(value)
    config.model.k_state = k_state
    print(k_state)
    
    return config