# ------------------------------------------------------------------------------------------------------------------- #
import os
import numpy as np
import random
import torch
import cv2
from torch.utils import data
from .transforms import get_affine_transform
from pathlib import Path
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
def get_im_path(root_dir, im_name):
    tkns = im_name.split('_')
    person = tkns[2]
    im_dir = os.path.join(root_dir, 'image', person) 
    if not os.path.isdir(im_dir):
        im_dir = os.path.join(root_dir, 'image', person) 
    im_path = os.path.join(im_dir, im_name+'.jpg')
    return im_path
# ------------------------------------------------------------------------------------------------------------------- #
def get_pred_img_path(root_dir:str, im_name):
    im_fpath = Path(root_dir).joinpath('image', im_name+'.jpg')
    if im_fpath.is_file():
        return im_fpath.as_posix()
    else:
        print(root_dir, im_fpath, im_name)
        return None
# ------------------------------------------------------------------------------------------------------------------- #
def get_anno_path(root_dir, im_name):
    tkns = im_name.split('_')
    person = tkns[2]
    anno_path = os.path.join(root_dir, 'label', person, im_name+'.png')
    return anno_path
# ------------------------------------------------------------------------------------------------------------------- #
def get_edge_path(root_dir, im_name):
    tkns = im_name.split('_')
    person = tkns[2]
    edge_path = os.path.join(root_dir, 'edge', person, im_name+'.png')
    return edge_path
# ------------------------------------------------------------------------------------------------------------------- #
def get_valid_list(root_dir, list_path, logger):
    availabe_list = []
    list_fpath = Path(list_path)
    cand_list = list_fpath.read_text().strip().split('\n')
    for im_name in cand_list:
        try:
            if 'pred' in list_fpath.name:
                im_fpath = get_pred_img_path(root_dir, im_name)
                im_fpath = Path(im_fpath) if im_fpath is not None else None
                if im_fpath is not None and im_fpath.is_file():
                    availabe_list.append(im_name)
                else:
                    print(im_fpath, im_name in cand_list)
                    logger.warning(f'  ~ any of image for <{im_name}> is not exist.')
            else:
                im_path = Path(get_im_path(root_dir, im_name))
                an_path = Path(get_anno_path(root_dir, im_name))
                ed_path = Path(get_edge_path(root_dir, im_name))
                if im_path.is_file() and an_path.is_file() and ed_path.is_file():
                    availabe_list.append(im_name)
                else:
                    logger.warning(f'  ~ any of image, label, edge for <{im_name}> is not exist.')
        except Exception as ex:
            logger.exception(f'{ex}')
    return availabe_list
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
class NIA2DataSet(data.Dataset):
    def __init__(self, logger, root, dataset_fpath, crop_size, scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, transform=None, bArgument=True):
        """
        :rtype:
        """
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.ignore_edge = 0 # in case of argumentation, border of edge should be set as 0 for ignore
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.transform = transform
        self.dataset_name = '' if dataset_fpath is None else dataset_fpath.name
        self.bArgument = bArgument
        list_path = dataset_fpath
        self.im_list = get_valid_list(self.root, list_path, logger) 
        self.number_samples = len(self.im_list)
    # --------------------------------------------------------------------------------------------------------------- #
    def __len__(self):
        return self.number_samples
    # --------------------------------------------------------------------------------------------------------------- #
    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)
    # --------------------------------------------------------------------------------------------------------------- #
    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale
    # --------------------------------------------------------------------------------------------------------------- #
    def __getitem__(self, index):
        # Load training image
        im_name = self.im_list[index]

        im_path = get_im_path(self.root, im_name) if 'pred' not in self.dataset_name else get_pred_img_path(self.root, im_name)
        org_anno_path = get_anno_path(self.root, im_name) if 'pred' not in self.dataset_name else None
        org_edge_path = get_edge_path(self.root, im_name) if 'pred' not in self.dataset_name else None
        
        if not os.path.isfile(im_path):
            self.logger.warning(f'  ==> {im_path} is not exist in dataloader')
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        h, w, _ = im.shape
        org_anno = np.zeros((h, w), dtype=np.long)

        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if 'train' in self.dataset_name or 'val' in self.dataset_name: 
            org_anno = cv2.imread(org_anno_path, cv2.IMREAD_GRAYSCALE)
            org_edge = cv2.imread(org_edge_path, cv2.IMREAD_GRAYSCALE)
            if 'train' in self.dataset_name and self.bArgument: 
                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                    if random.random() <= 0.6 else 0

        trans = get_affine_transform(center, s, r, self.crop_size) 
        input_im = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)) 

        if self.transform:
            input_im = self.transform(input_im)

        meta = {
            'name': im_name,
            'center': center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if 'test' in self.dataset_name or 'pred' in self.dataset_name:
            return input_im, [], [], meta
        elif 'val' in self.dataset_name:
            label_parsing = cv2.warpAffine(
                org_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(self.ignore_label)) 
            label_edge = cv2.warpAffine(
                org_edge,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(self.ignore_edge)) 
            label_parsing = torch.from_numpy(label_parsing)
            label_edge = torch.from_numpy(label_edge)
            return input_im, label_parsing, label_edge, meta
        elif 'train' in self.dataset_name: 
            label_parsing = cv2.warpAffine(
                org_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(self.ignore_label)) 
            label_edge = cv2.warpAffine(
                org_edge,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(self.ignore_edge)) 
                
            label_parsing = torch.from_numpy(label_parsing)
            label_edge = torch.from_numpy(label_edge)
            return input_im, label_parsing, label_edge, meta
    # --------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
