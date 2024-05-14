import os
import cv2
import numpy as np
import pandas as pd
import torch
import json
import random
from glob import glob
from PIL import Image
from utils.utils import load_json, load_image
from torch.utils.data import Dataset
from torchvision import transforms


class TileBatchDataset(Dataset):
    def __init__(self, 
                 tile_dir, 
                 mask_dir, 
                 tissue, 
                 cell_dir=None, 
                 deconv='POLARIS', 
                 prefix=['*'], 
                 channels=3, 
                 aug=True, 
                 val=False, 
                 panuke=None, 
                 sample_ratio=1, 
                 max_cell_num=128
                 ) -> None:
        ### Batch Size should be 1
        self.tile_list = []
        
        if aug is True:
            self.transforms = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.1, hue=0.2),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )
        self.channels = channels
        
        self.mask_dir = mask_dir
        self.cell_dir = cell_dir
        self.val = val
        self.panuke = panuke
        self.mcn = max_cell_num

        with open(tissue, 'r') as tissue_file:
            self.tc = json.load(tissue_file)
        print(f"Tissue Compartment: {self.tc['list']}")

        if cell_dir is not None:
            cell_labels = {}

            if deconv in ['Tangram', 'RCTD', 'stereoscope', 'CARD', 'POLARIS', 'Xenium', 'CARD_fix', 'RCTD_fix']:
                for cell_prop in glob(os.path.join(cell_dir, "*.tsv")):
                    sample_name = cell_prop.split('/')[-1].split('.')[0]
                    if sample_name not in prefix:
                        continue
                    cell_df = pd.read_csv(cell_prop, sep='\t', index_col=0)
                    for cell_index, row in cell_df.iterrows():
                        cell_index = str(cell_index)
                        x, y, barcode = cell_index.split('_')
                        cell_num = np.array(row)
                        cell_propotion = cell_num / np.sum(cell_num)
                        cell_labels.update(
                            {
                                sample_name + '_' + f'{x}x{y}': cell_propotion
                            }
                        )
                        abs_path = glob(os.path.join(tile_dir, sample_name, f'*_{x}x{y}.jpg'))
                        if len(abs_path) <= 0:
                            continue
                        img_name = abs_path[0].split('/')[-1].replace('.jpg', '')
                        json_file = os.path.join(self.mask_dir, sample_name, f"json/{img_name}.json")
                        if not os.path.exists(json_file):
                            continue
                        self.tile_list.append(abs_path[0])
            elif deconv in ['Mix']:
                for cell_prop in glob(os.path.join(cell_dir, "*.tsv")):
                    sample_name = cell_prop.split('/')[-1].replace('.tsv', '')
                    if sample_name not in prefix:
                        continue
                    cell_df = pd.read_csv(cell_prop, sep='\t', index_col=0)
                    for cell_index, row in cell_df.iterrows():
                        cell_index = "%06d" % int(cell_index)
                        abs_path = glob(os.path.join(tile_dir, sample_name, f'{cell_index}-Breast.jpg'))
                        if len(abs_path) == 0:
                            continue
                        cell_propotion = np.array(row)
                        cell_labels.update(
                            {
                                sample_name + '_' + cell_index + '-Breast': cell_propotion
                            }
                        )
                        img_name = abs_path[0].split('/')[-1].replace('.jpg', '')
                        json_file = os.path.join(self.mask_dir, sample_name, f"json/{img_name}.json")
                        if not os.path.exists(json_file):
                            continue
                        self.tile_list.append(abs_path[0])

            else:
                raise NotImplementedError("The Deconvolution Method is not supported.")

            self.cell_labels = cell_labels

            if not val:
                print(self.tc['dict'])
                print(cell_df.keys())
                assert len(self.tc['dict'].keys()) == len(cell_df.keys())
            print(f"Cell Category: {list(cell_df.keys())}")
        else:
            for dir_prefix in prefix:
                path_list = []
                if panuke is None:
                    panuke = '*'
                else:
                    panuke = f'*-{panuke}'
                for abs_path in glob(os.path.join(tile_dir, f"{dir_prefix}/{panuke}.jpg")):
                # for abs_path in glob(os.path.join(tile_dir, "tumor_003/*.jpg")):
                    img_name = abs_path.split('/')[-1].replace('.jpg', '')
                    sample_name = abs_path.split('/')[-2]
                    json_file = os.path.join(self.mask_dir, sample_name, f"json/{img_name}.json")
                    if os.path.exists(json_file):
                        path_list.append(abs_path)
                self.tile_list.extend(path_list)

        self.tile_list = random.sample(self.tile_list, int(len(self.tile_list) * sample_ratio))

    def __len__(self) -> int:
        return len(self.tile_list)

    def __getitem__(self, index: int):
        # load image
        img_path = self.tile_list[index]
        img_name = img_path.split('/')[-1].replace('.jpg', '')
        sample_name = img_path.split('/')[-2]
        with open(img_path, 'rb') as fp:
            pic = Image.open(fp)
            image = self.transforms(pic)
        
        # load mask
        json_file = os.path.join(self.mask_dir, sample_name, f"json/{img_name}.json")
        nucs = load_json(json_file)
        mask_list, size_list, pos_list, type_list = [], [], [], []
        for index, nuc in nucs.items():
            if len(image.shape) < 3:
                image = np.expand_dims(image, axis=0)
            # mask = np.zeros_like(image)
            center_x, center_y = nuc['centroid']
            bbox_xa, bbox_ya = nuc['bbox'][0]
            bbox_xb, bbox_yb = nuc['bbox'][1]
            mask = np.zeros([3, bbox_xb - bbox_xa, bbox_yb - bbox_ya])
            mask = image[:, bbox_xa: bbox_xb, bbox_ya: bbox_yb]
            try:
                mask = torch.tensor(cv2.resize(mask.permute(1, 2, 0).numpy(), (30, 30)))
            except:
                continue

            mask_list.append(mask.permute(2, 0, 1)), size_list.append(np.array([(bbox_xb - bbox_xa) / image.shape[-1], (bbox_yb - bbox_ya) / image.shape[-1]]))
            pos_list.append(np.array([center_x, center_y]))
            if nuc['type'] is None:
                type_list.append(-1)
            else:
                type_list.append(int(self.tc['HoVerNet'][int(nuc['type'])]))

        cell_types = np.array(type_list)
        image = torch.tensor(cv2.resize(image.permute(1, 2, 0).numpy(), (128, 128))).permute(2, 0, 1)
        cell_num = len(mask_list)
        if cell_num == 1:
            dist_mat = np.zeros([cell_num, cell_num])
            cell_coords = np.array([pos_list])
        elif cell_num == 0:
            dist_mat = np.zeros([cell_num, cell_num])
            cell_coords = np.zeros([0, 2])
        else:
            cell_coords = np.stack(pos_list, axis=0)
            dist_mat = np.zeros([cell_num, cell_num])
            for i in range(cell_num):
                for j in range(i + 1, cell_num):
                    dist = np.linalg.norm((cell_coords[i] - cell_coords[j]), ord=2)
                    if dist < 40:
                        dist_mat[i, j] = 1

        dist_mat = dist_mat + dist_mat.T + np.identity(cell_num)

        adj_mat = np.zeros([self.mcn, self.mcn])
        cell_pixels = np.zeros([self.mcn, 2])
        CellType = -np.ones([self.mcn])
        valid_mask = len(mask_list) if len(mask_list) < self.mcn else self.mcn  # max cell num
        if valid_mask >= self.mcn:
            mask_list = mask_list[:self.mcn]
            size_list = size_list[:self.mcn]
            adj_mat = dist_mat[:self.mcn, :self.mcn]
            cell_pixels = cell_coords[:self.mcn]
            CellType = cell_types[:self.mcn]
        else:
            for _ in range(self.mcn - valid_mask):
                mask_list.append(torch.zeros((3, 30, 30)))
                size_list.append(torch.zeros((2)))

            adj_mat = np.pad(dist_mat, ((0, self.mcn - valid_mask), (0, self.mcn - valid_mask)))
            cell_pixels[:valid_mask] = cell_coords
            CellType[:valid_mask] = cell_types

        cell_images = np.stack(mask_list, axis=0)
        cell_sizes = np.stack(size_list, axis=0)

        if self.cell_dir is None:
            return {
            'name': f'{sample_name}_{img_name}',
            'tissue': image,   # B 3 256 256
            'image': torch.tensor(cell_images, dtype=torch.float32),   # B 128 3 128 128
            'mask': int(valid_mask),    # B
            'size': torch.tensor(cell_sizes, dtype=torch.float32),
            'adj': torch.tensor(adj_mat, dtype=torch.long),
            'cell_coords': torch.tensor(cell_pixels, dtype=torch.float32),
            'cell_types': torch.tensor(CellType, dtype=torch.float32)
        }
        # Add tissue compartment
        if self.panuke is not None:
            sample_keys = sample_name + '_' + img_name.split('-')[0] + f'-{self.panuke}'  # Only for Panuke
            cell_prop = self.cell_labels[sample_keys]
            tissue_cat = -1

        else:
            sample_keys = sample_name + '_' + img_name.split('_')[-1]
            cell_prop = self.cell_labels[sample_keys]
            # tc = np.zeros(len(self.tc['list']))
            # for cata_i, prop in enumerate(cell_prop):
            #     tc[self.tc['list'].index(self.tc['dict'][str(cata_i)])] += prop
            if np.max(cell_prop) > 0.75:
                tissue_cat = self.tc['list'].index(self.tc['dict'][str(np.argmax(cell_prop))])
            else:
                tissue_cat = len(self.tc['list'])

        return {
            'name': f'{sample_name}_{img_name}',
            'tissue': image,   # B 3 128 128
            'image': torch.tensor(cell_images, dtype=torch.float32),   # B 128 3 128 128
            'mask': int(valid_mask),    # B
            'cells': torch.tensor(cell_prop, dtype=torch.float32),  # B cell_type
            'size': torch.tensor(cell_sizes, dtype=torch.float32),
            'adj': torch.tensor(adj_mat, dtype=torch.long),
            'tissue_cat': torch.tensor(tissue_cat, dtype=torch.long),    # B 1
            'cell_coords': torch.tensor(cell_pixels, dtype=torch.float32),
            'cell_types': torch.tensor(CellType, dtype=torch.float32)
        }
    

class TileBatchStateDataset(Dataset):
    def __init__(self, tile_dir, mask_dir, tissue, cell_dir=None, state_dir=None, deconv='POLARIS', prefix=['*'], channels=3, aug=True, val=False, panuke=False, sample_ratio=1, max_cell_num=128) -> None:
        ### Batch Size should be 1
        self.tile_list = []
        
        if aug is True:
            self.transforms = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.1, hue=0.2),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )
        self.channels = channels
        
        self.mask_dir = mask_dir
        self.cell_dir = cell_dir
        self.state_dir = state_dir
        self.val = val
        self.panuke = panuke
        self.mcn = max_cell_num

        with open(tissue, 'r') as tissue_file:
            self.tc = json.load(tissue_file)
        print(f"Tissue Compartment: {self.tc['list']}")

        if cell_dir is not None:
            cell_labels = {}
            state_labels = {}

            if deconv in ['Tangram', 'RCTD', 'stereoscope', 'CARD', 'POLARIS', 'Xenium']:
                for cell_prop in glob(os.path.join(cell_dir, "*.tsv")):
                    sample_name = cell_prop.split('/')[-1].split('.')[0]
                    if sample_name not in prefix:
                        continue
                    cell_df = pd.read_csv(cell_prop, sep='\t', index_col=0)
                    state_df = pd.read_csv(os.path.join(state_dir, f"{sample_name}.tsv"), sep='\t', index_col=0)
                    for cell_index, state_row in state_df.iterrows():
                        cell_index = str(cell_index)
                        x, y, barcode = cell_index.split('_')

                        # cell prop
                        row = cell_df.loc[cell_index]
                        cell_num = np.array(row)
                        cell_propotion = cell_num / np.sum(cell_num)
                        cell_labels.update(
                            {
                                sample_name + '_' + f'{x}x{y}': cell_propotion
                            }
                        )

                        # state prop
                        state_num = np.array(state_row)
                        state_proportion = state_num / np.sum(state_num)
                        state_labels.update(
                            {
                                sample_name + '_' + f'{x}x{y}': state_proportion
                            }
                        )

                        abs_path = glob(os.path.join(tile_dir, sample_name, f'*_{x}x{y}.jpg'))
                        if len(abs_path) <= 0:
                            continue
                        img_name = abs_path[0].split('/')[-1].split('.')[0]
                        json_file = os.path.join(self.mask_dir, sample_name, f"json/{img_name}.json")
                        if not os.path.exists(json_file):
                            continue
                        self.tile_list.append(abs_path[0])
            elif deconv in ['Mix']:
                for cell_prop in glob(os.path.join(cell_dir, "*.tsv")):
                    sample_name = cell_prop.split('/')[-1].split('.')[0]
                    if sample_name not in prefix:
                        continue
                    cell_df = pd.read_csv(cell_prop, sep='\t', index_col=0)
                    for cell_index, row in cell_df.iterrows():
                        cell_index = "%06d" % int(cell_index)
                        abs_path = glob(os.path.join(tile_dir, sample_name, f'{cell_index}-Prostate.jpg'))
                        if len(abs_path) == 0:
                            continue
                        cell_propotion = np.array(row)
                        cell_labels.update(
                            {
                                sample_name + '_' + cell_index + '-Prostate': cell_propotion
                            }
                        )
                        img_name = abs_path[0].split('/')[-1].split('.')[0]
                        json_file = os.path.join(self.mask_dir, sample_name, f"json/{img_name}.json")
                        if not os.path.exists(json_file):
                            continue
                        self.tile_list.append(abs_path[0])

            else:
                raise NotImplementedError("The Deconvolution Method is not supported.")

            self.cell_labels = cell_labels
            self.state_labels = state_labels

            if not val:
                print(self.tc['dict'])
                print(cell_df.keys())
                assert len(self.tc['dict'].keys()) == len(cell_df.keys())
            print(f"Cell Category: {list(cell_df.keys())}")
        else:
            for dir_prefix in prefix:
                path_list = []
                print(os.path.join(tile_dir, f"{dir_prefix}/*.jpg"))
                for abs_path in glob(os.path.join(tile_dir, f"{dir_prefix}/*.jpg")):
                    # img_name = abs_path.split('/')[-1].split('.')[0]
                    img_name = abs_path.split('/')[-1].strip('.jpg')
                    sample_name = abs_path.split('/')[-2]
                    json_file = os.path.join(self.mask_dir, sample_name, f"json/{img_name}.json")
                    # print(json_file)
                    if os.path.exists(json_file):
                        path_list.append(abs_path)
                self.tile_list.extend(path_list)

        self.tile_list = random.sample(self.tile_list, int(len(self.tile_list) * sample_ratio))
        
    def __len__(self) -> int:
        return len(self.tile_list)

    def __getitem__(self, index: int):
        # load image
        img_path = self.tile_list[index]
        img_name = img_path.split('/')[-1].strip('.jpg')
        sample_name = img_path.split('/')[-2]
        with open(img_path, 'rb') as fp:
            pic = Image.open(fp)
            image = self.transforms(pic)
        
        # load mask
        json_file = os.path.join(self.mask_dir, sample_name, f"json/{img_name}.json")
        nucs = load_json(json_file)
        mask_list, size_list, pos_list, type_list = [], [], [], []
        for index, nuc in nucs.items():
            if len(image.shape) < 3:
                image = np.expand_dims(image, axis=0)
            # mask = np.zeros_like(image)
            center_x, center_y = nuc['centroid']
            bbox_xa, bbox_ya = nuc['bbox'][0]
            bbox_xb, bbox_yb = nuc['bbox'][1]
            mask = np.zeros([3, bbox_xb - bbox_xa, bbox_yb - bbox_ya])
            mask = image[:, bbox_xa: bbox_xb, bbox_ya: bbox_yb]
            try:
                mask = torch.tensor(cv2.resize(mask.permute(1, 2, 0).numpy(), (30, 30)))
            except:
                continue

            mask_list.append(mask.permute(2, 0, 1)), size_list.append(np.array([(bbox_xb - bbox_xa) / image.shape[-1], (bbox_yb - bbox_ya) / image.shape[-1]]))
            pos_list.append(np.array([center_x, center_y]))
            type_list.append(int(self.tc['HoVerNet'][int(nuc['type'])]))

        cell_types = np.array(type_list)
        image = torch.tensor(cv2.resize(image.permute(1, 2, 0).numpy(), (256, 256))).permute(2, 0, 1)
        cell_num = len(mask_list)
        if cell_num <= 1:
            dist_mat = np.zeros([cell_num, cell_num])
            cell_coords = np.array(pos_list)
        else:
            cell_coords = np.stack(pos_list, axis=0)
            dist_mat = np.zeros([cell_num, cell_num])
            for i in range(cell_num):
                for j in range(i + 1, cell_num):
                    dist = np.linalg.norm((cell_coords[i] - cell_coords[j]), ord=2)
                    if dist < 40:
                        dist_mat[i, j] = 1

        dist_mat = dist_mat + dist_mat.T + np.identity(cell_num)

        adj_mat = np.zeros([self.mcn, self.mcn])
        cell_pixels = np.zeros([self.mcn, 2])
        CellType = np.zeros([self.mcn])
        valid_mask = len(mask_list) if len(mask_list) < self.mcn else self.mcn  # max cell num
        if valid_mask >= self.mcn:
            mask_list = mask_list[:self.mcn]
            size_list = size_list[:self.mcn]
            adj_mat = dist_mat[:self.mcn, :self.mcn]
            cell_pixels = cell_coords[:self.mcn, :self.mcn]
            CellType = cell_types[:self.mcn]
        else:
            for _ in range(self.mcn - valid_mask):
                mask_list.append(torch.zeros((3, 30, 30)))
                size_list.append(torch.zeros((2)))

            adj_mat = np.pad(dist_mat, ((0, self.mcn - valid_mask), (0, self.mcn - valid_mask)))
            for i in range(valid_mask):
                cell_pixels[i] = cell_coords[i]
            CellType[:valid_mask] = cell_types

        cell_images = np.stack(mask_list, axis=0)
        cell_sizes = np.stack(size_list, axis=0)

        if self.cell_dir is None:
            return {
            'name': f'{sample_name}_{img_name}',
            'tissue': image,   # B 3 128 128
            'image': torch.tensor(cell_images, dtype=torch.float32),   # B 128 3 128 128
            'mask': int(valid_mask),    # B
            'size': torch.tensor(cell_sizes, dtype=torch.float32),
            'adj': torch.tensor(adj_mat, dtype=torch.long),
            'cell_coords': torch.tensor(cell_pixels, dtype=torch.float32),
            'cell_types': torch.tensor(CellType, dtype=torch.float32)
        }
        # Add tissue compartment
        if self.panuke:
            sample_keys = sample_name + '_' + img_name.split('-')[0] + '-Prostate'  # Only for Panuke
            cell_prop = self.cell_labels[sample_keys]
            tissue_cat = -1

        else:
            sample_keys = sample_name + '_' + img_name.split('_')[-1]
            cell_prop = self.cell_labels[sample_keys]
            # tc = np.zeros(len(self.tc['list']))
            # for cata_i, prop in enumerate(cell_prop):
            #     tc[self.tc['list'].index(self.tc['dict'][str(cata_i)])] += prop
            if np.max(cell_prop) > 0.75:
                tissue_cat = self.tc['list'].index(self.tc['dict'][str(np.argmax(cell_prop))])
            else:
                tissue_cat = len(self.tc['list'])

            state_prop = self.state_labels[sample_keys]

        return {
            'name': f'{sample_name}_{img_name}',
            'tissue': image,   # B 3 128 128
            'image': torch.tensor(cell_images, dtype=torch.float32),   # B 128 3 128 128
            'mask': int(valid_mask),    # B
            'cells': torch.tensor(cell_prop, dtype=torch.float32),  # B cell_type
            'states': torch.tensor(state_prop, dtype=torch.float32),
            'size': torch.tensor(cell_sizes, dtype=torch.float32),
            'adj': torch.tensor(adj_mat, dtype=torch.long),
            'tissue_cat': torch.tensor(tissue_cat, dtype=torch.long),    # B 1
            'cell_coords': torch.tensor(cell_pixels, dtype=torch.float32),
            'cell_types': torch.tensor(CellType, dtype=torch.float32)
        }