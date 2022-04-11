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
import os
import torchvision


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


def normalise(x):
    return (x - x.min()) / max(x.max() - x.min(), 0.0001)


def adjust_label_order_for_wandb(x):
    x = x.copy()
    x += 1
    x[x == NUM_CLASSES] = 0
    return x


def hierarchical_perturbation(model, input, interp_mode='nearest', resize=None, perturbation_type='mean',
                              threshold_mode='var', return_info=False, diff_func=torch.relu, max_depth=-1, verbose=True,
                              cell_init=2):
    if verbose: print('\nBelieve the HiPe!')
    with torch.no_grad():
        dev = input.device
        print('Using device: {}'.format(dev))
        bn, channels, input_y_dim, input_x_dim = input.shape
        dim = min(input_x_dim, input_y_dim)
        total_masks = 0
        depth = 0
        num_cells = int(max(np.ceil(np.log2(dim)), 1) / cell_init)
        base_max_depth = int(np.log2(dim / num_cells)) - 2
        if max_depth == -1 or max_depth > base_max_depth + 2:
            max_depth = base_max_depth
        if verbose: print('Max depth: {}'.format(max_depth))
        saliency = torch.zeros((1, NUM_CLASSES, input_y_dim, input_x_dim), device=dev)

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
            if threshold_mode == 'var':
                threshold = torch.amin(saliency, dim=(-1, -2)) + (
                        (torch.amax(saliency, dim=(-1, -2)) - torch.amin(saliency, dim=(-1, -2))) / 2)
                threshold = -torch.var(threshold)
            elif threshold_mode == 'mean':
                threshold = torch.mean(saliency)
            else:
                threshold = torch.min(saliency) + ((torch.max(saliency) - torch.min(saliency)) / 2)

            print('Threshold: {}'.format(threshold))
            thresholds_d_list.append(diff_func(threshold))

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
                        if threshold_mode == 'var':
                            local_saliency = -torch.var(torch.amax(local_saliency, dim=(-1, -2)))
                        else:
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
                if perturbation_type != 'fade':
                    b_imgs = b_list.pop()
                masks = masks_list.pop()

                # resize low-res masks to input size
                masks = F.interpolate(masks, (input_y_dim, input_x_dim), mode=interp_mode)

                if perturbation_type == 'fade':
                    perturbed_outputs = diff_func(output - model(input * masks)[0][0])
                else:
                    perturbed_outputs = diff_func(output - model(b_imgs)[0][0])

                if len(list(perturbed_outputs.shape)) == 1:
                    sal = perturbed_outputs.reshape(-1, 1, 1, 1) * torch.abs(masks - 1)
                else:
                    sal = perturbed_outputs.reshape(1, NUM_CLASSES, 1, 1) * torch.abs(masks - 1)

                saliency += sal

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
        step_size = k_size
    x_steps = range(0, input_x_dim - k_size + 1, step_size)
    y_steps = range(0, input_y_dim - k_size + 1, step_size)
    heatmap = torch.zeros((NUM_CLASSES, len(y_steps), len(x_steps)))
    num_occs = 0

    blur_substrate = blur(input)
    hx = 0
    for x in x_steps:
        hy = 0
        for y in y_steps:
            print('{}/{}'.format(num_occs, len(x_steps) * len(y_steps)))
            occ_im = input.clone()

            if args.perturbation_type == 'mean':
                occ_im[:, :, y: y + k_size, x: x + k_size] = torch.mean(input[:, :, y: y + k_size, x: x + k_size],
                                                                        axis=(-1, -2), keepdims=True)
            if args.perturbation_type == 'fade':
                occ_im[:, :, y: y + k_size, x: x + k_size] = 0.0
            if args.perturbation_type == 'blur':
                occ_im[:, :, y: y + k_size, x: x + k_size] = blur_substrate[:, :, y: y + k_size, x: x + k_size]

            diff = torch.relu(output - model(occ_im)[0][0]).reshape(NUM_CLASSES, 1, 1)
            heatmap[:, hy:hy + 1, hx:hx + 1] += diff
            num_occs += 1
            hy += 1
        hx += 1

    return heatmap, num_occs


class ModelUmbrella(nn.Module):

    def __init__(self, feature_extractor, inf_model):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.inf_model = inf_model

    def forward(self, x):
        return self.inf_model(self.feature_extractor(x))


def sort_coords(coords, centre):
    # sort coordinates by chebyshev distance from centre coord
    centre = centre.split(',')
    x, y = int(centre[1]), int(centre[0])
    print('Sorting patches around {},{}'.format(x, y))
    coords = list(coords)
    coords.sort(key=lambda p: max(abs(x - p[1]), abs(y - p[0])))
    return coords


def overlap_coords(coords, overlap):
    olc = []
    for c in coords:
        o = [c[0] + overlap, c[1]]
        if o not in olc: olc.append(o)
        o = [c[0] - overlap, c[1]]
        if o not in olc: olc.append(o)
        o = [c[0], c[1] + overlap]
        if o not in olc: olc.append(o)
        o = [c[0], c[1] - overlap]
        if o not in olc: olc.append(o)

    coords.extend(olc)

    return coords


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saliency segmentation script')
    parser.add_argument('--slide_path', type=str, default='../heatmaps/demo/slides/', help='path to isyntax slide')
    parser.add_argument('--slide_name', type=str, default='IC-EN-00266-01', help='path to isyntax slide')
    parser.add_argument('--ckpt_path', type=str, default='../heatmaps/demo/ckpts/s_0_checkpoint.pt',
                        help='path to model checkpoint')
    parser.add_argument('--patch_path', type=str, default='../heatmaps/demo/patches/patches/',
                        help='path to h5 patch file')
    parser.add_argument('--txt_path', type=str, default='../annotations/', help='path to txt annotation files')
    parser.add_argument('--max_patches', type=int, default=100, help='Number of patches to extract and segment')
    parser.add_argument('--cell_init', type=int, default=2, help='HiPe cell initialisation hyperparameter.')
    parser.add_argument('--max_depth', type=int, default=1, help='Hierarchical perturbation depth. Higher is '
                                                                 'more detailed but takes much longer.')
    parser.add_argument('--perturbation_type', default='fade', help='Perturbation substrate for use in '
                                                                    'hierarchical perturbation.')
    parser.add_argument('--interp_mode', default='bicubic', help='Interpolation mode for up/downsampling')
    parser.add_argument('--downsample', type=int, default=8, help='Downsample for final image and saliency '
                                                                  'segmentation stitching. 1 = no downsampling. If '
                                                                  'all patches are used, low values will probably '
                                                                  'cause OOM errors when the final image is stitched.')
    parser.add_argument('--save_high_res_patches', default=False, action='store_true', help='Whether to save high '
                                                                                            'resolution patches and '
                                                                                            'saliency segmentations '
                                                                                            'to WandB log before '
                                                                                            'stitching.')
    parser.add_argument('--use_flat_perturbation', default=False, action='store_true', help='Whether to use flat '
                                                                                            'peturbation instead of '
                                                                                            'HiPe. Sightly faster at '
                                                                                            'high kernel sizes, '
                                                                                            'but much slower and lacks '
                                                                                            'relative '
                                                                                            'saliency detail at low '
                                                                                            'kernel sizes.')
    parser.add_argument('--flat_kernel_size', type=int, default=32, help='Kernel size for flat perturbation.')
    parser.add_argument('--centre', default='45000,45000', help='Coordinate in form x,y of central patch')
    parser.add_argument('--threshold_mode', default='var', help='HiPe threshold mode')
    parser.add_argument('--save_path', default='', help='where to save saliency segmentation png file. If empty, '
                                                        'no local save is used. All images are logged to WandB in any '
                                                        'case.')
    parser.add_argument('--overlap', default=False, action='store_true', help='Overlap patches')
    parser.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing saliency '
                                                                                'segmentation patches, if they exist')

    args = parser.parse_args()

    print(args)

    proj = "icaird_sal_seg"
    run = wandb.init(project=proj, entity="jessicamarycooper", config=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load models
    model_args = argparse.Namespace(
            **{'model_type': 'clam_sb', 'model_size': 'small', 'drop_out': 'true', 'n_classes': 3})
    label_list = ['malignant', 'insufficient', 'other_benign']
    NUM_CLASSES = len(label_list)
    wandb_class_labels = {0: 'other_benign', 1: 'malignant', 2: 'insufficient'}

    inf_model = initiate_model(model_args, args.ckpt_path).to(device)
    inf_model.eval()
    feature_extractor = resnet50_baseline(pretrained=True).to(device)
    feature_extractor.eval()
    model = ModelUmbrella(feature_extractor, inf_model)

    # load WSI
    wsi = WholeSlideImage(args.slide_path + args.slide_name + '.isyntax')
    transforms = default_transforms()

    args_code = '-'.join([str(s) for s in
                          [args.slide_name, args.max_depth, args.perturbation_type, args.interp_mode, args.downsample,
                           args.use_flat_perturbation, args.flat_kernel_size, args.threshold_mode, args.cell_init]])
    print(args_code)

    # create sal_seg dir
    if not os.path.exists('sal_seg'): os.mkdir('sal_seg')

    # create run dir
    if not os.path.exists('sal_seg/{}'.format(args_code)): os.mkdir('sal_seg/{}'.format(args_code))

    # load patch data
    with h5py.File(args.patch_path + args.slide_name + '.h5', 'r') as f:
        coords = f['coords']
        patch_level = coords.attrs['patch_level']
        patch_size = coords.attrs['patch_size']

        _, ydim, _, xdim = wsi.level_dimensions[patch_level]
        min_x, min_y, max_x, max_y = xdim, ydim, 0, 0

        pdim = patch_size // args.downsample

        max_patches = len(coords) if args.max_patches == -1 or args.max_patches > len(coords) else args.max_patches
        wandb.log({
            'Patch Level': patch_level, 'Patch Size': patch_size, 'Num Patches': max_patches, 'Slide': args.slide_name
            })

        coords = sort_coords(coords, centre=args.centre)[:max_patches]
        if args.overlap:
            coords = overlap_coords(coords, patch_size // 2)
            max_patches = len(coords)
        print('Generating patch-level saliency...')
        for i, coord in enumerate(coords):
            print('{}/{} Patch coords: {}'.format(i + 1, max_patches, coord))

            if (not args.overwrite) and os.path.exists('sal_seg/{}/sal_seg_{}'.format(args_code, coord)):
                print('Found existing saliency segmentation patch for coord {}, skipping...'.format(coord))
            else:
                img = transforms(wsi.read_region(RegionRequest(coord, patch_level, (patch_size, patch_size)))).to(
                        device)
                logits, Y_prob, Y_hat, A_raw, results_dict = model(torch.Tensor(img.unsqueeze(0)))
                logits = np.round(logits.detach().numpy(), 2)[0]

                if args.use_flat_perturbation:
                    sal_maps, _ = flat_perturbation(model, img.unsqueeze(0), k_size=args.flat_kernel_size)

                else:
                    sal_maps, _ = hierarchical_perturbation(model, img.unsqueeze(0),
                                                            perturbation_type=args.perturbation_type,
                                                            interp_mode=args.interp_mode, verbose=True,
                                                            max_depth=args.max_depth,
                                                            threshold_mode=args.threshold_mode,
                                                            cell_init=args.cell_init)

                torch.save(F.interpolate(img.unsqueeze(0), (pdim, pdim), mode=args.interp_mode)[0],
                           'sal_seg/{}/img_{}'.format(args_code, coord))
                torch.save(F.interpolate(sal_maps.unsqueeze(0), (pdim, pdim), mode=args.interp_mode)[0],
                           'sal_seg/{}/sal_seg_{}'.format(args_code, coord))
                if args.save_high_res_patches:
                    max_seg = torch.argmax(sal_maps, dim=0).int()
                    min_seg = torch.argmin(sal_maps, dim=0).int()
                    sal_seg = torch.where((min_seg != max_seg), max_seg, torch.zeros_like(max_seg) + NUM_CLASSES)
                    sal_maps = normalise(sal_maps)

                    wandb.log({
                        'Prediction': label_list[torch.argmax(Y_prob)],
                        'Saliency': [wandb.Image(sal_maps[n], caption=label_list[n]) for n in range(NUM_CLASSES)],
                        'Blended Saliency': wandb.Image(sal_maps),
                        'Saliency Segmentation': wandb.Image(img, caption=str(logits), masks={
                            "predictions": {
                                "mask_data"   : adjust_label_order_for_wandb(sal_seg.numpy()),
                                "class_labels": wandb_class_labels
                                }
                            })
                        })

            y, x = coord
            if x < min_x: min_x = x
            if y < min_y: min_y = y
            x1, y1 = x + patch_size, y + patch_size
            if x1 > max_x: max_x = x1
            if y1 > max_y: max_y = y1

        im_x, im_y = max_x - min_x, max_y - min_y
        im_x, im_y = im_x // args.downsample, im_y // args.downsample

        print('Full image size: {}x{}'.format(im_x, im_y))

        full_img = torch.ones((3, im_x, im_y))
        full_sal_map = torch.zeros((NUM_CLASSES, im_x, im_y))

        print('Stitching...')

        for i, coord in enumerate(coords):
            print('{}/{}'.format(i + 1, max_patches))
            img = torch.load('sal_seg/{}/img_{}'.format(args_code, coord))
            sal_maps = torch.load('sal_seg/{}/sal_seg_{}'.format(args_code, coord))
            y, x = coord
            x = (x - min_x) // args.downsample
            y = (y - min_y) // args.downsample

            if img.shape[1] != pdim:
                img = F.interpolate(img.unsqueeze(0), (pdim, pdim))[0]
                sal_maps = F.interpolate(sal_maps.unsqueeze(0), (pdim, pdim))[0]

            full_img[:, x: x + pdim, y:y + pdim] = img
            full_sal_map[:, x:x + pdim, y:y + pdim] = torch.maximum(sal_maps, full_sal_map[:, x:x + pdim, y:y + pdim])

        print('Calculating saliency segmentation...')
        max_seg = torch.argmax(full_sal_map, dim=0).int()
        min_seg = torch.argmin(full_sal_map, dim=0).int()
        full_sal_seg = torch.where((min_seg != max_seg), max_seg, torch.zeros_like(max_seg) + NUM_CLASSES)
        full_sal_map = normalise(full_sal_map)

        print('Logging images...')
        wandb.log({
            'Image dimensions'                                                         : [im_x, im_y],
            'Region coords'                                                            : [min_x, max_x, min_y, max_y],
            'Saliency'                                                                 : [
                wandb.Image(full_sal_map[n], caption=label_list[n]) for n in range(NUM_CLASSES)],
            'Full Blended Saliency'                                                    : wandb.Image(full_sal_map),
            'Full Saliency Segmentation'                                               : wandb.Image(full_img, masks={
                "predictions": {
                    "mask_data"   : adjust_label_order_for_wandb(full_sal_seg.int().numpy()),
                    "class_labels": wandb_class_labels
                    }
                })
            })

        if len(args.save_path) > 0:
            Image.fromarray(full_sal_seg.numpy()).save(
                args.save_path + '_saliency_segmentation_' + args.slide_name + '.png')

        if len(args.txt_path) > 0:
            print('Evaluating segmentation performance...')
            wsi.initXML(args.txt_path + args.slide_name + '.txt')
            annotation = wsi.visWSI(vis_level=0, color=(0, 255, 0), hole_color=(0, 0, 255), annot_color=(255, 0, 0),
                                     line_thickness=12, max_size=None, top_left=None, bot_right=None,
                                     custom_downsample=args.downsample, view_slide_only=False, number_contours=False,
                                     seg_display=False, annot_display=True)

            print(annotation.shape)
            print(full_img.shape)
            annot = annotation[:,min_x//args.downsample:max_x//args.downsample,
                         min_y//args.downsample:max_y//args.downsample]

            print(annot)

            print(annot.shape)
            print()
            annot[annot != 1.0] = 0.0

            malignant_ss = full_sal_seg
            malignant_ss[malignant_ss != 0] = 1
            malignant_ss = torch.abs(malignant_ss - 1).float()

            malignant_an = annot[0]

            output = malignant_ss
            target = malignant_an

            tp = torch.sum(target * output)
            tn = torch.sum((1 - target) * (1 - output))
            fp = torch.sum((1 - target) * output)
            fn = torch.sum(target * (1 - output))

            p = tp / (tp + fp + 0.0001)
            r = tp / (tp + fn + 0.0001)
            f1 = 2 * p * r / (p + r + 0.0001)
            acc = (tp + tn) / (tp + tn + fp + fn)
            dice = (2 * tp) / (2*tp + fp + fn)

            wandb.log({'Expert': wandb.Image(malignant_an), 'Machine': wandb.Image(malignant_ss), 'Precision':p,
                       'Recall':r,
                       'F1':f1,
            'Accuracy':acc,
                       'Dice':dice})


        print('Done!')

    run.finish()
