import torch
import os
import argparse
from model.arch import HistoState
from data import TileBatchStateDataset
from utils.utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from configs import _get_cell_state_config

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser(description="Prediction with Spots Images")
    parser.add_argument('--seed', default=47, type=int)
    parser.add_argument('--model', default='Baseline', type=str, help='model description')
    parser.add_argument('--tissue', default='BRCA', type=str)
    parser.add_argument('--deconv', default='CARD', type=str)
    parser.add_argument('--subtype', action='store_true')
    parser.add_argument('--prefix', required = False, nargs = '+')
    parser.add_argument('--tissue_compartment', type=str, required=True)

    args = parser.parse_args()
    config = _get_cell_state_config(args.tissue, args.deconv, args.subtype, args.tissue_compartment)

    setup_seed(args.seed)
    print(f"Model details: {args.model}")
    
    print("Load Dataset...")

    data_prefix = args.prefix if isinstance(args.prefix, list) else [args.prefix]

    print(data_prefix)
    train_data = TileBatchStateDataset(config.data.tile_dir, config.data.mask_dir, config.data.tissue_dir, config.data.cell_dir, config.data.state_dir, deconv=config.data.deconv, prefix=data_prefix)
    print(f"length of train data: {len(train_data)}")
    train_loader = DataLoader(train_data, batch_size=config.data.batch_size, shuffle=True, num_workers=6, pin_memory=True)
    
    print("Load Model...")
    model = HistoState(config.model)
    loss_func = torch.nn.KLDivLoss()
    aux_loss = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.train.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using 1 gpu")
    else:
        print("Using cpu")
    

    model_dir, ckpt_dir = \
        os.path.join(config.data.save_model, args.model), os.path.join(config.data.ckpt, args.model)
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

    print("Training with Cell Types...")
    if current_epoch + 1 <= config.train.epoch:
        for iter in range(current_epoch + 1, config.train.epoch):
            train_loop(iter, model, train_loader, loss_func, aux_loss, optimizer, device)
            
            if iter % config.train.val_iter == 0 and iter > config.train.val_min_iter:
                save_checkpoint(model, optimizer, os.path.join(model_dir, f'epoch_{iter}.ckpt'))

    print("Training with Cell States...")
    for iter in range(max(config.train.epoch, current_epoch + 1), config.train.epoch + config.train.state_epoch):
        train_loop(iter, model, train_loader, loss_func, aux_loss, optimizer, device, with_state=True)
        
        if iter % config.train.val_iter == 0 and iter > config.train.val_min_iter:
            save_checkpoint(model, optimizer, os.path.join(model_dir, f'epoch_{iter}.ckpt'))
            
    print("Training finished!")
    print(f"Results saved in {os.path.join(config.data.ckpt, args.model)}")


def train_loop(epoch, model, train_loader, loss_func, aux_loss, opt, device, with_state=False):
    model.train()
    train_bar = tqdm(train_loader, desc="epoch " + str(epoch), total=len(train_loader),
                            unit="batch", dynamic_ncols=True)
    for idx, data in enumerate(train_bar):
        tissue = data['tissue'].to(torch.float32).to(device)
        images, cell_proportion, state_proportion = \
            data['image'].to(torch.float32).to(device), data['cells'].to(torch.float32).to(device), data['states'].to(torch.float32).to(device)
        valid_mask = data['mask'].to(torch.long).to(device)
        cell_size = data['size'].to(torch.float32).to(device)
        adj_mat = data['adj'].to(torch.float32).to(device)
        gt_tissue_cat = data['tissue_cat'].to(device)
        if torch.sum(data['mask']) <= 0:
            continue
        batch, cells, channels, height, width = images.shape
        images = images.reshape(batch * cells, channels, height, width)
        pred_dict = model(tissue, images, adj_mat, cell_size, valid_mask, {'batch': batch, 'cells': cells})
        cell_prop_list = []
        for single_props, valid_index in zip(cell_proportion, valid_mask):
            if valid_index <= 0:
                continue
            cell_prop_list.append(single_props)
        cell_proportion = torch.stack(cell_prop_list, dim=0)

        loss_out = loss_func((cell_proportion + 1e-10).log(), pred_dict['type_prop'] + 1e-10) + \
                loss_func((pred_dict['type_prop'] + 1e-10).log(), cell_proportion + 1e-10) + \
                aux_loss(pred_dict['tissue_compartment'], gt_tissue_cat)
        
        if with_state:
            state_prop_list = []
            for name, single_states, valid_index in zip(data['name'], state_proportion, valid_mask):
                if valid_index <= 0:
                    continue
                state_prop_list.append(single_states)
            state_proportion = torch.stack(state_prop_list, dim=0)

            state_loss = loss_func((state_proportion + 1e-10).log(), pred_dict['state_prop'] + 1e-10) + loss_func((pred_dict['state_prop'] + 1e-10).log(), state_proportion + 1e-10)
            # consist_loss = torch.nn.functional.l1_loss(pred_dict['state_prop'][:, :5], pred_dict['type_prop'][:, [0, 2, 3, 5, 6]])
            consist_loss = torch.nn.functional.l1_loss(pred_dict['state_prop'][:, :6], pred_dict['type_prop'][:, :6])
            loss_out = loss_out + state_loss + 0.5 * consist_loss

        loss_out.backward()
        opt.step()
        opt.zero_grad()

        loss_value = loss_out.item()
        train_bar.set_description('epoch:{} iter:{} loss:{}'.format(epoch, idx, loss_value))

if __name__ == '__main__':
    main()