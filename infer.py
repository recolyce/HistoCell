import torch
import os
import argparse
import pickle as pkl
from model.arch import HistoCell
from data import TileBatchDataset
from utils.utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs import _get_config

# def _get_config(tissue_type, deconv, subtype, k_class, tissue_dir):
#     config = CN()
#     config.train = CN()
#     config.train.lr = 0.0005
#     config.train.epoch = 41
#     config.train.val_iter = 10
#     config.train.val_min_iter = 9

#     config.data = CN()
#     config.data.deconv = deconv
#     config.data.save_model = f'./train_log/{tissue_type}/add_type/models'
#     config.data.ckpt = f'./train_log/{tissue_type}/add_type/ckpts'
#     config.data.tile_dir = f'/HOME/scz5693/run/data/{tissue_type}/tiles'
#     config.data.mask_dir = f'/HOME/scz5693/run/hover_net/cls_results/{tissue_type}'
#     config.data.batch_size = 64
#     config.data.tissue_dir = tissue_dir
#     config.data.max_cell_num = 256

#     config.model = CN()
#     config.model.tissue_class = 3
#     config.model.pretrained = True
#     config.model.channels = 3

#     if subtype:
#         config.data.cell_dir = f'/HOME/scz5693/run/data/{tissue_type}/cell_proportion/subtype/{config.data.deconv}'
        
#     else:
#         config.data.cell_dir = f'/HOME/scz5693/run/data/{tissue_type}/cell_proportion/type/{config.data.deconv}'

#     config.model.k_class = k_class
    
#     return config

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser(description="Prediction with Spots Images")
    parser.add_argument('--seed', default=47, type=int)
    parser.add_argument('--model', default='Baseline', type=str, help='model description')
    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--tissue', default='BRCA', type=str)
    parser.add_argument('--deconv', default='CARD', type=str)
    parser.add_argument('--subtype', action='store_true')
    parser.add_argument('--prefix', required=False, nargs = '+')
    parser.add_argument('--k_class', default=8, type=int)
    parser.add_argument('--tissue_compartment', type=str, required=True)
    parser.add_argument('--omit_gt', action='store_true')
    parser.add_argument('--val_panuke', default=None, type=str)

    args = parser.parse_args()
    config = _get_config(args.tissue, args.deconv, args.subtype, args.k_class, args.tissue_compartment)

    print(f"Model details: {args.model}")
    
    print("Load Dataset...")

    data_prefix = args.prefix if isinstance(args.prefix, list) else [args.prefix]

    print(data_prefix)

    val_data = TileBatchDataset(config.data.tile_dir, config.data.mask_dir, config.data.tissue_dir, cell_dir=None, 
                                deconv=config.data.deconv, prefix=data_prefix, 
                                aug=False, val=args.omit_gt, panuke=args.val_panuke, max_cell_num=config.data.max_cell_num)
    print(f"length of train data: {len(val_data)}")
    # val_data = TileMaskDataset(config.data.tile_dir, config.data.mask_dir, config.data.cell_dir)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=8, pin_memory=True)
    # val_loader = DataLoader(val_data, batch_size=1, collate_fn=collate_fn)
    
    print("Load Model...")
    model = HistoCell(config.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using 1 gpu")
    else:
        print("Using cpu")
    

    model_dir, ckpt_dir = \
        os.path.join(config.data.save_model, args.model), os.path.join(config.data.ckpt, args.model)
    os.makedirs(model_dir, exist_ok=True), os.makedirs(ckpt_dir, exist_ok=True)
    
    ckpt_file = find_ckpt(model_dir)

    if args.epoch >= 0:
        state_dict, current_epoch = load_ckpt(os.path.join(model_dir, f"epoch_{args.epoch}.ckpt"))
        model.load_state_dict(state_dict)
    
    # ipdb.set_trace()
    elif ckpt_file is not None:
        state_dict, current_epoch = load_ckpt(os.path.join(model_dir, ckpt_file))
        model.load_state_dict(state_dict)
    
    else:
        raise FileNotFoundError("No trained model exits")

    model.to(device)
    
    print(f"Current Epoch: {current_epoch}")

    val_results = val_loop(current_epoch, model, val_loader, device)
    with open(os.path.join(ckpt_dir, f'epoch_{current_epoch}_{data_prefix[0]}_val.pkl'), 'wb') as file:
        pkl.dump(val_results, file)
            
    print("Val Finished!")
    print(f"Results saved in {os.path.join(config.data.ckpt, args.model)}")


def val_loop(epoch, model, val_loader, device):
    model.train()
    val_bar = tqdm(val_loader, desc="epoch " + str(epoch), total=len(val_loader),
                            unit="batch", dynamic_ncols=True)
    
    all_results = {}
    for idx, data in enumerate(val_bar):
        tissue = data['tissue'].to(torch.float32).to(device)
        images = data['image'].to(torch.float32).to(device)
        cell_size = data['size'].to(torch.float32).to(device)
        adj_mat = data['adj'].to(torch.float32).to(device)
        valid_mask = data['mask'].to(torch.long).to(device)
        cell_coords = data['cell_coords'].to(torch.float32).to(device)
        ref_type = data['cell_types'].to(torch.long).to(device)
        if torch.sum(data['mask']) <= 0:
            continue
        batch, cells, channels, height, width = images.shape
        images = images.reshape(batch * cells, channels, height, width)
        prob_list, pred_proportion, tissue_cat, cell_embeddings = model(tissue, images, adj_mat, cell_size, valid_mask, {'batch': batch, 'cells': cells})
        name_list, valid_list, coord_list, type_list = [], [], [], []
        for name, valid_index, valid_coord, valid_type in zip(data['name'], valid_mask, cell_coords, ref_type):
            if valid_index <= 0:
                continue
            name_list.append(name)
            valid_list.append(valid_index)
            coord_list.append(valid_coord)
            type_list.append(valid_type)

        for name, pb, pp, vix, cc, ct, cell_features in zip(name_list, prob_list, pred_proportion, valid_list, coord_list, type_list, cell_embeddings):
            valid_num = int(vix.detach().cpu())
            all_results.update(
                {
                    name: {
                        'prob': pb.detach().cpu().numpy(),
                        'pred_proportion': pp.detach().cpu().numpy(),
                        'prior_type': ct.cpu().numpy(),
                        'cell_num': valid_num,
                        'cell_coords': cc[:valid_num].detach().cpu().numpy(),
                        # 'cell_embedding': cell_features.detach().cpu().numpy()
                    }
                }
            )

        val_bar.set_description('epoch:{} iter:{}'.format(epoch, idx))

    return all_results


if __name__ == '__main__':
    main()