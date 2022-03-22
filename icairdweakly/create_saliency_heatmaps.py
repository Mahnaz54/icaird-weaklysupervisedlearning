from __future__ import print_function

import numpy as np

import math
import argparse
import cv2
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models.resnet_custom import resnet50_baseline
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches, hierarchical_perturbation
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.WholeSlideImage import RegionRequest
from datasets.wsi_dataset import Wsi_Region
from create_patches import seg_and_patch
from datasets.wsi_dataset import default_transforms

import wandb

parser = argparse.ArgumentParser(description='Saliency segmentation script')
parser.add_argument('--slide_path', type=str, default='../heatmaps/demo/slides/IC-EN-00033-01.isyntax',
                    help='path to isyntax slide')
parser.add_argument('--ckpt_path', type=str, default='../heatmaps/demo/ckpts/s_0_checkpoint.pt',
                    help='path to model checkpoint')
parser.add_argument('--patch_path', type=str, default='../heatmaps/demo/patches/patches/IC-EN-00033-01.h5',
                    help='path to model checkpoint')
parser.add_argument('--level', type=int, default=6)
parser.add_argument('--max_patches', type=int, default=-1)
parser.add_argument('--hipe_max_depth', type=int, default=2)
parser.add_argument('--hipe_perturbation_type', default='mean')

args = parser.parse_args()

print(args)

proj = "icaird_sal_seg"
run = wandb.init(project=proj, entity="jessicamarycooper", config=args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelUmbrella(nn.Module):

    def __init__(self, feature_extractor, inf_model):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.inf_model = inf_model

    def forward(self, x):
        return self.inf_model(self.feature_extractor(x))


# load models
model_args = argparse.Namespace(**{'model_type': 'clam_sb', 'model_size': 'small', 'drop_out': 'true', 'n_classes': 3})
label_list = ['malignant', 'insufficient', 'other_benign']
num_classes = len(label_list)
inf_model = initiate_model(model_args, args.ckpt_path)
inf_model.eval()
feature_extractor = resnet50_baseline(pretrained=True)
feature_extractor.eval()
model = ModelUmbrella(feature_extractor, inf_model)

# load WSI
wsi = WholeSlideImage(args.slide_path, hdf5_file=None)
transforms = default_transforms()

# get overall prediction # TODO

# load patch data
with h5py.File(args.patch_path, 'r') as f:
    coords = f['coords']
    patch_level = coords.attrs['patch_level']
    patch_size = coords.attrs['patch_size']
    for i, coord in enumerate(coords):
        if i == args.max_patches:
            run.finish()
            exit()
        img = transforms(wsi.read_region(RegionRequest(coord, patch_level, (patch_size, patch_size))))
        logits, Y_prob, Y_hat, A_raw, results_dict = model(torch.Tensor(img.unsqueeze(0)))
        logits = np.round(logits.detach().numpy(), 2)[0]
        print(i, logits)
        hipe_maps = []
        for c in range(num_classes):
            hipe_maps.append(
                hierarchical_perturbation(model, img.unsqueeze(0), c, perturbation_type=args.hipe_perturbation_type,
                                          verbose=True,
                                          max_depth=args.hipe_max_depth)[0])
        wandb.log({
                      'Patch'.format(i): wandb.Image(img, caption=str(logits)),
                      'HiPe'           : [wandb.Image(hipe_maps[h], caption=label_list[h]) for h in range(num_classes)]
                  })

# for each patch, get saliency map

# stitch patches and saliency map

run.finish()
