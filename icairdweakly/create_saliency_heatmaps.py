from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.nn as nn
from utils.utils import *
from utils.eval_utils import initiate_model as initiate_model
from models.resnet_custom import resnet50_baseline
import h5py
from wsi_core.WholeSlideImage import WholeSlideImage, RegionRequest
from datasets.wsi_dataset import default_transforms
import torch.nn.functional as F
import wandb
from scipy.ndimage.filters import gaussian_filter
from PIL import Image


def gkern(klen, nsig):
    inp = np.zeros((klen, klen))
    inp[klen // 2, klen // 2] = 1
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))


def blur(x, klen=11, ksig=5):
    kern = gkern(klen, ksig)
    return F.conv2d(x, kern, padding=klen // 2)


def hierarchical_perturbation(model, input, interp_mode='nearest', resize=None, batch_size=1,
                              perturbation_type='mean', threshold_mode='mid-range', return_info=False,
                              diff_func=torch.relu, max_depth=-1, verbose=True):
    if verbose: print('\nBelieve the HiPe!')
    with torch.no_grad():
        dev = input.device
        print('Using device: {}'.format(dev))
        if dev == 'cpu':
            batch_size = 1
        bn, channels, input_y_dim, input_x_dim = input.shape
        dim = min(input_x_dim, input_y_dim)
        total_masks = 0
        depth = 0
        num_cells = int(max(np.ceil(np.log2(dim)), 1) / 2)
        base_max_depth = int(np.log2(dim / num_cells)) - 2
        if max_depth == -1 or max_depth > base_max_depth + 2:
            max_depth = base_max_depth
        if verbose: print('Max depth: {}'.format(max_depth))
        saliency = torch.zeros((1, num_classes, input_y_dim, input_x_dim), device=dev)
        max_batch = batch_size

        thresholds_d_list = []
        masks_d_list = []

        output = model(input)[0]

        if perturbation_type == 'blur':
            pre_b_image = blur(input.clone().cpu()).to(dev)

        while depth < max_depth:
            masks_list = []
            b_list = []
            num_cells *= 2
            depth += 1
            if threshold_mode == 'mean':
                threshold = torch.mean(saliency)
            else:
                threshold = torch.min(saliency) + ((torch.max(saliency) - torch.min(saliency)) / 2)

            thresholds_d_list.append(diff_func(threshold).item())

            y_ixs = range(-1, num_cells)
            x_ixs = range(-1, num_cells)
            x_cell_dim = input_x_dim // num_cells
            y_cell_dim = input_y_dim // num_cells

            if verbose:
                print('Depth: {}, {} x {} Cell Dim'.format(depth, y_cell_dim, x_cell_dim))
            possible_masks = 0

            for x in x_ixs:
                for y in y_ixs:
                    possible_masks += 1
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(x + 2, num_cells), min(y + 2, num_cells)

                    mask = torch.zeros((1, 1, num_cells, num_cells), device=dev)
                    mask[:, :, y1:y2, x1:x2] = 1.0
                    local_saliency = F.interpolate(mask, (input_y_dim, input_x_dim), mode=interp_mode) * saliency

                    if depth > 1:
                        local_saliency = torch.max(diff_func(local_saliency))
                    else:
                        local_saliency = 0

                    # If salience of region is greater than the average, generate higher resolution mask
                    if local_saliency >= threshold:
                        masks_list.append(abs(mask - 1))

                        if perturbation_type == 'blur':
                            b_image = input.clone()
                            b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim,
                            x1 * x_cell_dim:x2 * x_cell_dim] = pre_b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim,
                                                               x1 * x_cell_dim:x2 * x_cell_dim]
                            b_list.append(b_image)

                        if perturbation_type == 'mean':
                            b_image = input.clone()
                            mean = torch.mean(
                                    b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim],
                                    axis=(-1, -2), keepdims=True)

                            b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim] = mean
                            b_list.append(b_image)

            num_masks = len(masks_list)
            if verbose: print('Selected {}/{} masks at depth {}'.format(num_masks, possible_masks, depth))
            if num_masks == 0:
                depth -= 1
                break
            total_masks += num_masks
            masks_d_list.append(num_masks)

            while len(masks_list) > 0:
                m_ix = min(len(masks_list), max_batch)
                if perturbation_type != 'fade':
                    b_imgs = torch.cat(b_list[:m_ix])
                    del b_list[:m_ix]
                masks = torch.cat(masks_list[:m_ix])
                del masks_list[:m_ix]

                # resize low-res masks to input size
                masks = F.interpolate(masks, (input_y_dim, input_x_dim), mode=interp_mode)

                if perturbation_type == 'fade':
                    perturbed_outputs = diff_func(output - model(input * masks)[0])
                else:
                    perturbed_outputs = diff_func(output - model(b_imgs)[0])

                if len(list(perturbed_outputs.shape)) == 1:
                    sal = perturbed_outputs.reshape(-1, 1, 1, 1) * torch.abs(masks - 1)
                else:
                    sal = perturbed_outputs * torch.abs(masks - 1)

                saliency += torch.sum(sal, dim=(0, 1))

        if verbose: print('Used {} masks in total.'.format(total_masks))
        if resize is not None:
            saliency = F.interpolate(saliency, (resize[1], resize[0]), mode=interp_mode)
        if return_info:
            return saliency[0], {'thresholds': thresholds_d_list, 'masks': masks_d_list, 'total_masks': total_masks}
        else:
            return saliency[0], total_masks


def flat_perturbation(model, input, k_size=1, step_size=-1):
    bn, channels, input_y_dim, input_x_dim = input.shape
    output = model(input)[0]
    if step_size == -1:
        step_size = k_size//2
    x_steps = range(0, input_x_dim - k_size + 1, step_size)
    y_steps = range(0, input_y_dim - k_size + 1, step_size)
    heatmap = torch.zeros((1, num_classes, input_y_dim, input_x_dim))
    num_occs = 0
    for x in x_steps:
        for y in y_steps:
            print('{}/{}'.format(num_occs, len(x_steps) * len(y_steps)))
            occ_im = input.clone()
            occ_im[:, :, y: y + k_size, x: x + k_size] = torch.mean(input[:, :, y: y + k_size, x: x + k_size],
                                    axis=(-1, -2), keepdims=True)

            print(torch.relu(output - model(occ_im)[0]).shape)
            heatmap[:,:, y:y+k_size, x:x+k_size] += torch.relu(output - model(occ_im)[0][0])
            num_occs += 1

    return heatmap[0], num_occs


class ModelUmbrella(nn.Module):

    def __init__(self, feature_extractor, inf_model):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.inf_model = inf_model

    def forward(self, x):
        return self.inf_model(self.feature_extractor(x))


def sort_coords(coords):
    coords = list(coords)
    coords.sort(key=lambda p: p[0] + p[1])
    return coords


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saliency segmentation script')
    parser.add_argument('--slide_path', type=str, default='../heatmaps/demo/slides/IC-EN-00033-01.isyntax',
                        help='path to isyntax slide')
    parser.add_argument('--ckpt_path', type=str, default='../heatmaps/demo/ckpts/s_0_checkpoint.pt',
                        help='path to model checkpoint')
    parser.add_argument('--patch_path', type=str, default='../heatmaps/demo/patches/patches/IC-EN-00033-01.h5',
                        help='path to h5 patch file')
    parser.add_argument('--max_patches', type=int, default=-1, help='Number of patches to extract and segment')
    parser.add_argument('--hipe_max_depth', type=int, default=1, help='Hierarchical perturbation depth. Higher is '
                                                                      'more detailed but takes much longer.')
    parser.add_argument('--hipe_perturbation_type', default='mean', help='Perturbation substrate for use in '
                                                                         'hierarchical perturbation.')
    parser.add_argument('--hipe_interp_mode', default='nearest', help='Interpolation mode for hierarchical '
                                                                      'perturbation')
    parser.add_argument('--downsample', type=int, default=4, help='Downsample for final image and saliency '
                                                                  'segmentation stitching. 1 = no downsampling. If '
                                                                  'all patches are used, low values will probably '
                                                                  'cause OOM errors.')
    parser.add_argument('--save_high_res_patches', default=False, action='store_true', help='Whether to save high '
                                                                                            'resolution patches and '
                                                                                            'saliency segmentations '
                                                                                            'to WandB log before '
                                                                                            'stitching.')
    parser.add_argument('--use_flat_perturbation', default=False, action='store_true', help='Whether to use flat '
                                                                                            'peturbation instead of '
                                                                                            'HiPe. Sightly faster at '
                                                                                            'high kernel sizes, '
                                                                                            'but lacks relative '
                                                                                            'saliency detail.')
    parser.add_argument('--flat_kernel_size', type=int, default=32, help='Kernel size for flat perturbation.')
    parser.add_argument('--save_path', default='', help='where to save saliency segmentation png file. If empty, '
                                                        'no local save is used. All images are logged to WandB in any case.')

    args = parser.parse_args()

    print(args)

    proj = "icaird_sal_seg"
    run = wandb.init(project=proj, entity="jessicamarycooper", config=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load models
    model_args = argparse.Namespace(
        **{'model_type': 'clam_sb', 'model_size': 'small', 'drop_out': 'true', 'n_classes': 3})
    label_list = ['malignant', 'insufficient', 'other_benign']
    num_classes = len(label_list)
    class_labels = dict(zip(range(0, num_classes), label_list))

    inf_model = initiate_model(model_args, args.ckpt_path).to(device)
    inf_model.eval()
    feature_extractor = resnet50_baseline(pretrained=True).to(device)
    feature_extractor.eval()
    model = ModelUmbrella(feature_extractor, inf_model)

    # load WSI
    wsi = WholeSlideImage(args.slide_path)
    transforms = default_transforms()
    # load patch data
    with h5py.File(args.patch_path, 'r') as f:
        coords = f['coords']
        patch_level = coords.attrs['patch_level']
        patch_size = coords.attrs['patch_size']
        slide_name = args.slide_path.split('/')[-1].split('.')[0]
        _, ydim, _, xdim = wsi.level_dimensions[patch_level]
        xdim, ydim = xdim // args.downsample, ydim // args.downsample
        min_x, min_y, max_x, max_y = xdim, ydim, 0, 0

        pdim = patch_size // args.downsample
        all_imgs = []
        all_sal_segs = []
        all_coords = []

        max_patches = len(coords) if args.max_patches == -1 or args.max_patches > len(coords) else args.max_patches
        wandb.log({
                      'Patch Level': patch_level, 'Patch Size': patch_size, 'Num Patches': max_patches,
                      'Slide'      : slide_name
                  })

        coords = sort_coords(coords)
        print('Generating patch-level saliency...')
        for i, coord in enumerate(coords):
            if i == max_patches:
                break
            img = transforms(wsi.read_region(RegionRequest(coord, patch_level, (patch_size, patch_size)))).to(device)
            logits, Y_prob, Y_hat, A_raw, results_dict = model(torch.Tensor(img.unsqueeze(0)))
            logits = np.round(logits.detach().numpy(), 2)[0]
            print('{}/{} Patch coords: {} Logits: {}'.format(i + 1, max_patches, coord, logits))

            if args.use_flat_perturbation:
                sal_maps = flat_perturbation(model, img.unsqueeze(0), k_size=args.flat_kernel_size)

            else:
                sal_maps = hierarchical_perturbation(model, img.unsqueeze(0),
                                                              perturbation_type=args.hipe_perturbation_type,
                                                              interp_mode=args.hipe_interp_mode, verbose=True,
                                                              max_depth=args.hipe_max_depth)

            sal_seg = torch.argmax(sal_maps, dim=1).int()[0]
            all_imgs.append(img)
            all_sal_segs.append(sal_seg)
            if args.save_high_res_patches:
                wandb.log({
                    'Prediction'       : label_list[torch.argmax(Y_prob)],
                    'Saliency'             : [wandb.Image(sal_maps[h], caption=label_list[h]) for h in range(
                            num_classes)],
                    'Saliency Segmentation': wandb.Image(img, caption=str(logits), masks={
                        "predictions": {
                            "mask_data": sal_seg.numpy(), "class_labels": class_labels
                            }
                        })
                    })

            y, x = coord // args.downsample
            if x < min_x: min_x = x
            if y < min_y: min_y = y
            x1, y1 = x + pdim, y + pdim
            if x1 > max_x: max_x = x1
            if y1 > max_y: max_y = y1

            all_coords.append((x, x1, y, y1))

        im_x, im_y = max_x - min_x, max_y - min_y

        print('Full image size: {}x{}'.format(im_x, im_y))

        full_img = torch.ones((3, im_x, im_y))
        full_sal_seg = torch.zeros((im_x, im_y)) + num_classes

        print('Stitching...')
        print(all_coords)

        for i in range(len(all_imgs)):
            print('{}/{}'.format(i + 1, len(all_imgs)))
            img = all_imgs[i]
            sal_seg = all_sal_segs[i]
            x, x1, y, y1 = all_coords[i]
            x, x1, y, y1 = x - min_x, x1 - min_x, y - min_y, y1 - min_y

            full_img[:, x: x1, y:y1] = F.interpolate(img.unsqueeze(0), (pdim, pdim))[0]
            full_sal_seg[x:x1, y:y1] = F.interpolate(sal_seg.float().unsqueeze(0).unsqueeze(0), (pdim, pdim))[0][0]

        wandb.log({
            'Region coords': [min_x * args.downsample, max_x * args.downsample, min_y * args.downsample,
                              max_y * args.downsample],
            'Full Saliency Segmentation': wandb.Image(full_img, masks={
                "predictions": {
                    "mask_data": full_sal_seg.int().numpy(), "class_labels": class_labels
                    }
                })
            })

        if len(args.save_path) > 0:
            Image.fromarray(full_sal_seg.numpy()).save(args.save_path + '_saliency_segmentation_' + slide_name + '.png')
        print('Done!')

    run.finish()
