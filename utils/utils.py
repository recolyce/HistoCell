import numpy as np
import json
import torch
import random
import os
import cv2
import scipy
from scipy.optimize import linear_sum_assignment
from PIL import Image

def reg_loss(props: torch.Tensor):
    # props: torch.Size([Batch Cell Prob])
    log_prop = torch.log(props)
    loss = props.mul(log_prop)
    return -torch.mean(loss)

def index_filter(used_idx, all_idx):
    rest_index = []
    for item in all_idx:
        if item in used_idx:
            continue
        rest_index.append(item)
    return rest_index

def load_image(img_path):
    fp = open(img_path, 'rb')
    pic = Image.open(fp)
    pic = np.array(pic)
    fp.close()
    return pic

def save_image(src, img_path):
    img = Image.fromarray(src, 'RGB')
    img.save(img_path)

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        nuc_info = json.load(file)

    return nuc_info['nuc']

def dump_json(src: dict, json_path):
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(src, file)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def collate_fn(batch):
    tensor_batch = {}
    for key, value in batch[0].items():
        if isinstance(value, str):
            tensor_batch.update({key: value})
            continue
        tensor_batch.update({key: torch.tensor(value, dtype=torch.float32)})

    return tensor_batch

def collate_batch_fn(iter_batch):
    tensor_batch = {key: [] for key in iter_batch[0].keys()}
    for batch in iter_batch:
        for key, value in batch.items():
            tensor_batch[key].append(value)
        
    for key in tensor_batch.keys():
        if key in ['tissue', 'cells', 'image']:
            tensor_batch[key] = torch.stack(tensor_batch[key], dim=0).to(torch.float32)
        if key in ['mask']:
            tensor_batch[key] = torch.tensor(tensor_batch[key], dtype=torch.long)

    return tensor_batch

def find_ckpt(file_dir):
    list=os.listdir(file_dir)
    list.sort(key=lambda fn: os.path.getmtime(os.path.join(file_dir, fn)) if not os.path.isdir(os.path.join(file_dir+fn)) else 0)
    if list != []:
        return list[-1]
    else:
        return None


def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    model = ckpt['state_dict']
    current_epoch = int(path.split('/')[-1].split('.')[0].split('_')[-1])
    return model, current_epoch

def save_checkpoint(model, optimizer, save_dir):
    print(f"Saving checkpoint to {save_dir}")
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_dir)

def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def get_centroid(pred_inst, types = None, label_dict = None, id_offset = 0):
    inst_id_list = np.unique(pred_inst[pred_inst > 0])  # exlcude background 0411 is cancer v.s. normal
    inst_info_dict = {}
    for inst_id in inst_id_list:
        inst_map = pred_inst == inst_id
        # TODO: chane format of bbox output
        rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
        inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
        inst_map = inst_map[
            inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
        ]
        inst_map = inst_map.astype(np.uint8)
        inst_moment = cv2.moments(inst_map)
        inst_contour = cv2.findContours(
            inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # * opencv protocol format may break
        inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
        # < 3 points dont make a contour, so skip, likely artifact too
        # as the contours obtained via approximation => too small or sthg
        if inst_contour.shape[0] < 3:
            continue
        if len(inst_contour.shape) != 2:
            continue # ! check for trickery shape
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid = np.array(inst_centroid)
        inst_contour[:, 0] += inst_bbox[0][1]  # X
        inst_contour[:, 1] += inst_bbox[0][0]  # Y
        inst_centroid[0] += inst_bbox[0][1]  # X
        inst_centroid[1] += inst_bbox[0][0]  # Y
        inst_info_dict[inst_id + id_offset] = {  # inst_id should start at 1
            "bbox": inst_bbox,
            "centroid": inst_centroid,
            "contour": inst_contour,
            "type": types if label_dict is None else label_dict[types],
        }

    return inst_info_dict


def pair_coordinates(setA, setB, radius):
    """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal 
    unique pairing (largest possible match) when pairing points in set B 
    against points in set A, using distance as cost function.

    Args:
        setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points 
        radius: valid area around a point in setA to consider 
                a given coordinate in setB a candidate for match
    Return:
        pairing: pairing is an array of indices
        where point at index pairing[0] in set A paired with point
        in set B at index pairing[1]
        unparedA, unpairedB: remaining poitn in set A and set B unpaired

    """
    # * Euclidean distance as the cost matrix
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric='euclidean')

    # * Munkres pairing with scipy library
    # the algorithm return (row indices, matched column indices)
    # if there is multiple same cost in a row, index of first occurence 
    # is return, thus the unique pairing is ensured
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

    # extract the paired cost and remove instances 
    # outside of designated radius
    pair_cost = pair_distance[indicesA, paired_indicesB]

    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]

    pairing = np.concatenate([pairedA[:,None], pairedB[:,None]], axis=-1)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
    return pairing, unpairedA, unpairedB

def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred