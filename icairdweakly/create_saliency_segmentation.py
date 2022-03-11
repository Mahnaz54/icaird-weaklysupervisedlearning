import wandb
import numpy as np
import torch.optim as optim
import pandas as pd
import os
import argparse
import random
import torchvision.transforms.functional as TF
import torch
from utils.utils import *
from datasets.dataset_generic import Generic_MIL_Dataset
from models.model_clam import CLAM_SB, CLAM_MB
from models.model_mil import MIL_fc, MIL_fc_mc
from models.resnet_custom import resnet50_baseline

pd.set_option('display.max_columns', None)


################################################ SET UP SEED AND ARGS AND EXIT HANDLER


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(0)

import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='../results/Endometrial/Results_Feb2022/patch_256/CLAM_sb/s_0_checkpoint.pt')
parser.add_argument('--img_path', default='../../../mnt/isilon1/iCAIRD/')
parser.add_argument('--model_type', default='CLAM_SB')
parser.add_argument('--dim', default=256, type=int)


def normalise(x):
    return (x - x.min()) / max(x.max() - x.min(), 0.0001)


args = parser.parse_args()

print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["WANDB_SILENT"] = "true"

proj = "saliency_segmentation"
run = wandb.init(project=proj, entity="jessicamarycooper", config=args)

labels = ['malignant', 'insufficient', 'other_benign']

att_model = eval(args.model_type)(n_classes=len(labels))

ckpt_path = args.ckpt
ckpt = torch.load(ckpt_path, map_location='cpu')

ckpt_clean = {}
for key in ckpt.keys():
    ckpt_clean.update({key.replace('3', '2'): ckpt[key]})

att_model.load_state_dict(ckpt_clean)
att_model.relocate()
att_model.eval()
feature_model = resnet50_baseline(pretrained=True).to(device)
feature_model.eval()


def model(x):
    return att_model(feature_model(x))[0]


# Load a slide and convert to tensor


# Split slide into patches


# Get model prediction for each patch

# Get saliency map for each patch

# Stitch map and patches.
