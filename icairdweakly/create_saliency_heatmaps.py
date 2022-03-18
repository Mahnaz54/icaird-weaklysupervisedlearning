from __future__ import print_function

import numpy as np

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

import wandb

parser = argparse.ArgumentParser(description='Saliency segmentation script')
parser.add_argument('--slide_path', type=str, default='../heatmaps/demo/slides/IC-EN-00033-01.isyntax',
                    help='path to isyntax slide')
parser.add_argument('--ckpt_path', type=str, default='../heatmaps/demo/ckpts/s_0_checkpoint.pt',
                    help='path to model checkpoint')
args = parser.parse_args()

proj = "icaird_sal_seg"
run = wandb.init(project=proj, entity="jessicamarycooper", config=args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelUmbrella(nn.Module):

    def __init__(self, feature_extractor, inf_model):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.inf_model = inf_model

    def forward(self, x):
        return self.model(self.feature_extractor(x))


# load models
model_args = argparse.Namespace(**{'model_type': 'clam_sb', 'model_size': 'small', 'drop_out': 'true', 'n_classes':3})
inf_model = initiate_model(model_args, args.ckpt_path)
feature_extractor = resnet50_baseline(pretrained=True)
feature_extractor.eval()
model = ModelUmbrella(feature_extractor, inf_model)

# load slide
print('Loading WSI...')
wsi = WholeSlideImage(args.slide_path)
print('Segmenting WSI...')
wsi.segmentTissue()
print('Visualising WSI...')
img = wsi.visWSI()

wandb.log({'Image':img})
# get patches from slide


# for each patch, get saliency map

#stitch patches and saliency map

run.finish()