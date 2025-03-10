import torch
import numpy as np
import os
import argparse
import random
import pickle as pkl
from model.arch import HistoCell
from data import TileBatchDataset
from utils.utils import *
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm
from yacs.config import CfgNode as CN
from torch.optim import Adam

def _get_config(tissue_type, deconv, subtype, k_class, tissue_dir):
    config = CN()
    config.train = CN()
    config.train.lr = 0.0005
    config.train.epoch = 61
    config.train.val_iter = 20
    config.train.val_min_iter = 1

    config.data = CN()
    config.data.deconv = deconv
    config.data.save_model = f'./train_log/pannuke/{tissue_type}/models'
    config.data.ckpt = f'./train_log/pannuke/{tissue_type}/ckpts'
    config.data.tile_dir = f'/data/gcf22/PaNnuke/raw_images'
    config.data.mask_dir = f'/data/gcf22/PaNnuke/cls_results_cv'
    config.data.batch_size = 64
    config.data.tissue_dir = tissue_dir

    config.model = CN()
    config.model.tissue_class = 3
    config.model.pretrained = True
    config.model.channels = 3

    config.data.cell_dir = f'/data/gcf22/PaNnuke/cell_props'

    config.model.k_class = k_class
    
    return config


def train_loop(epoch, model, train_loader, loss_func, aux_loss, opt, device):
    model.train()
    train_bar = tqdm(train_loader, desc="epoch " + str(epoch), total=len(train_loader),
                            unit="batch", dynamic_ncols=True)
    for idx, data in enumerate(train_bar):
        tissue = data['tissue'].to(torch.float32).to(device)
        images, cell_proportion = data['image'].to(torch.float32).to(device), data['cells'].to(torch.float32).to(device)
        valid_mask = data['mask'].to(torch.long).to(device)
        cell_size = data['size'].to(torch.float32).to(device)
        adj_mat = data['adj'].to(torch.float32).to(device)
        gt_tissue_cat = data['tissue_cat'].to(device)
        if torch.sum(data['mask']) <= 0:
            continue
        batch, cells, channels, height, width = images.shape
        images = images.reshape(batch * cells, channels, height, width)
        probs, pred_proportion, tissue_cat, _ = model(tissue, images, adj_mat, cell_size, valid_mask, {'batch': batch, 'cells': cells})
        cell_prop_list = []
        for single_props, valid_index in zip(cell_proportion, valid_mask):
            if valid_index <= 0:
                continue
            cell_prop_list.append(single_props)
        cell_proportion = torch.stack(cell_prop_list, dim=0)

        loss_out = loss_func((cell_proportion + 1e-10).log(), pred_proportion + 1e-10) + \
                loss_func((pred_proportion + 1e-10).log(), cell_proportion + 1e-10)
        loss_out.backward()
        opt.step()
        opt.zero_grad()

        loss_value = loss_out.item()
        train_bar.set_description('epoch:{} iter:{} loss:{}'.format(epoch, idx, loss_value))


def val_loop(epoch, model, val_loader, device):
    model.train()
    val_bar = tqdm(val_loader, desc="epoch " + str(epoch), total=len(val_loader),
                            unit="batch", dynamic_ncols=True)
    
    all_results = {}
    for idx, data in enumerate(val_bar):
        tissue = data['tissue'].to(torch.float32).to(device)
        images, cell_proportion = data['image'].to(torch.float32).to(device), data['cells'].to(torch.float32).to(device)
        cell_size = data['size'].to(torch.float32).to(device)
        adj_mat = data['adj'].to(torch.float32).to(device)
        valid_mask = data['mask'].to(torch.long).to(device)
        cell_coords = data['cell_coords'].to(torch.float32).to(device)
        ref_type = data['cell_types'].to(torch.long).to(device)
        if torch.sum(data['mask']) <= 0:
            continue
        batch, cells, channels, height, width = images.shape
        images = images.reshape(batch * cells, channels, height, width)
        prob_list, pred_proportion, _, cell_embeddings = model(tissue, images, adj_mat, cell_size, valid_mask, {'batch': batch, 'cells': cells})
        name_list, valid_list, coord_list, type_list, cell_prop_list = [], [], [], [], []
        for name, single_prop, valid_index, valid_coord, valid_type in zip(data['name'], cell_proportion, valid_mask, cell_coords, ref_type):
            if valid_index <= 0:
                continue
            name_list.append(name)
            valid_list.append(valid_index)
            coord_list.append(valid_coord)
            type_list.append(valid_type)
            cell_prop_list.append(single_prop)

        for name, pb, pp, cp, vix, cc, ct in zip(name_list, prob_list, pred_proportion, cell_prop_list, valid_list, coord_list, type_list):
            valid_num = int(vix.detach().cpu())
            all_results.update(
                {
                    name: {
                        'pred_proportion': pp.detach().cpu().numpy(),
                        'cell_num': valid_num,
                        'cell_coords': cc[:valid_num].detach().cpu().numpy(),
                        'prob': pb.detach().cpu().numpy(),
                        'prior_type': ct[:valid_num].detach().cpu().numpy(),
                        'gt_proportion': cp.detach().cpu().numpy(),
                        # 'cell_embedding': cell_features.detach().cpu().numpy()
                    }
                }
            )
        val_bar.set_description('epoch:{} iter:{}'.format(epoch, idx))
    return all_results


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser(description="Prediction with Spots Images" )
    parser.add_argument('--seed', default=47, type=int)
    parser.add_argument('--model', default='Baseline', type=str, help='model description')
    parser.add_argument('--tissue', default='*', type=str)
    parser.add_argument('--deconv', default='CARD', type=str)
    parser.add_argument('--subtype', action='store_true')
    parser.add_argument('--prefix', required = False, nargs = '+')
    parser.add_argument('--k_class', default=8, type=int)
    parser.add_argument('--tissue_compartment', type=str, required=True)
    parser.add_argument('--sample_ratio', default = 1, type=float)
    parser.add_argument('--reso', default = 1, type=int)

    args = parser.parse_args()
    config = _get_config(args.tissue, args.deconv, args.subtype, args.k_class, args.tissue_compartment)

    setup_seed(args.seed)
    print(f"Model details: {args.model}")
    
    print("Load Dataset...")
    data_prefix = args.prefix if isinstance(args.prefix, list) else [args.prefix]
    print(data_prefix)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using 1 gpu")
    else:
        print("Using cpu")

    for idx, val_prefix in enumerate(data_prefix):
        raw_prefix = [item for item in data_prefix]
        raw_prefix.pop(idx)
        val_fold = TileBatchDataset(config.data.tile_dir, config.data.mask_dir, config.data.tissue_dir, config.data.cell_dir, deconv=config.data.deconv, prefix=[val_prefix], focus_tissue=args.tissue)
        train_fold = TileBatchDataset(config.data.tile_dir, config.data.mask_dir, config.data.tissue_dir, config.data.cell_dir, deconv=config.data.deconv, prefix=raw_prefix, focus_tissue=args.tissue, reso=args.reso)
        print(f"length of train data: {len(train_fold)} with sample {raw_prefix}")
        train_fold,_ = torch.utils.data.random_split(train_fold, [int(len(train_fold)*args.sample_ratio), len(train_fold) - int(len(train_fold)*args.sample_ratio)])

        print(f"length of sampled train data: {len(train_fold)} with sample {raw_prefix}")
        print(f"length of val data: {len(val_fold)} with sample {val_prefix}")

        train_loader = DataLoader(train_fold, batch_size=config.data.batch_size, num_workers=6, pin_memory=True)
        val_loader = DataLoader(val_fold, batch_size=config.data.batch_size // 2, num_workers=6, pin_memory=True)
        
        print("Load Model...")
        model = HistoCell(config.model)
        loss_func = torch.nn.KLDivLoss()
        aux_loss = torch.nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=config.train.lr)

        model_dir, ckpt_dir = \
            os.path.join(config.data.save_model, args.model, val_prefix), os.path.join(config.data.ckpt, args.model, val_prefix)
        os.makedirs(model_dir, exist_ok=True), os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_file = find_ckpt(model_dir)
        # ipdb.set_trace()
        if ckpt_file is not None:
            state_dict, current_epoch = load_ckpt(os.path.join(model_dir, ckpt_file))
            model.load_state_dict(state_dict)
        else:
            current_epoch = -1

        model.to(device)
        
        print(f"Current Epoch: {current_epoch}")

        print("Training...")
        for iter in range(current_epoch + 1, config.train.epoch):
            train_loop(iter, model, train_loader, loss_func, aux_loss, optimizer, device)
            # if iter % 5 == 0:
            #     save_checkpoint(model, optimizer, os.path.join(model_dir, f'epoch_{iter}.ckpt'))
            # TODO
            if iter > config.train.val_min_iter and iter % config.train.val_iter == 0:
                print("##### Validation #####")
                val_results = val_loop(iter, model, val_loader, device)
                save_checkpoint(model, optimizer, os.path.join(model_dir, f'epoch_{iter}.pt'))
                with open(os.path.join(ckpt_dir, f'epoch_{iter}_val.pkl'), 'wb') as file:
                    pkl.dump(val_results, file)

                # save_checkpoint(model, optimizer, os.path.join(model_dir, f'epoch_{iter}.ckpt'))
                
        print("Training finished!")
        print(f"Results saved in {os.path.join(config.data.ckpt, args.model)}")

if __name__ == '__main__':
    main()