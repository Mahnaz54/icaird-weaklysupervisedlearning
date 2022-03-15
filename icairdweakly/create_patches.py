import os
import math
import time
import pdb
import argparse

import numpy as np
import pandas as pd

from wsi_core.WholeSlideImage import WholeSlideImage, get_best_level_for_downsample
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df


def segment(WSI_object, seg_params, filter_params):
    start_time = time.time()
    WSI_object.segmentTissue(**seg_params, filter_params=filter_params)
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed


def patching(WSI_object, **kwargs):
    start_time = time.time()
    file_path = WSI_object.process_contours(**kwargs)
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def stitching(file_path, wsi_object, downscale=64):
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
    total_time = time.time() - start
    return heatmap, total_time


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, patch_size=256, step_size=256,
                  seg_params={
                      'seg_level'  : -1, 'sthresh': 10, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': 'none',
                      'exclude_ids': 'none'
                  }, filter_params={'a_t': 100, 'a_h': 16, 'max_n_holes': 20},
                  vis_params={'vis_level': -1, 'line_thickness': 500},
                  patch_params={'use_padding': True, 'contour_fn': 'four_pt'}, patch_level=0, use_default_params=False,
                  seg=False, save_mask=True, stitch=False, patch=False, auto_skip=True, process_list=None):
    # List th slides to be processed
    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if 'isyntax' in slide and os.path.isfile(os.path.join(source, slide))]
    # Iitialize the dataframe with the name of slides and parameters 
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

    mask = df['process'] == 1
    process_stack = df[mask]
    total = len(process_stack)

    legacy_support = 'a' in df.keys()
    if legacy_support:
        print('detected legacy segmentation csv file, legacy support enabled')
        df = df.assign(**{
            'a_t'           : np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
            'a_h'           : np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
            'max_n_holes'   : np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
            'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
            'contour_fn'    : np.full((len(df)), patch_params['contour_fn'])
        })

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    # Start processing slides in order using the process_list file
    for i in range(total):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)

        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        print("\n\nprogress: {:.2f}, {}/{}".format(i / total, i, total))
        print('processing {}'.format(slide))

        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide)

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Inialize WSI
        full_path = os.path.join(source, slide)
        WSI_object = WholeSlideImage(full_path)

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()

        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}

            for key in vis_params.keys():
                if legacy_support and key == 'vis_level':
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                if legacy_support and key == 'a_t':
                    old_area = df.loc[idx, 'a']
                    seg_level = df.loc[idx, 'seg_level']
                    scale = WSI_object.level_downsamples(seg_level)
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (32 * 32))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                if legacy_support and key == 'seg_level':
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})
        # If slide is scanned with just one zoon level then set the visualization level to zero, else downsample the
        # slide to level 6
        img_lvl = len(WSI_object.level_dimensions) - 1

        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dimensions) == 1:
                current_vis_params['vis_level'] = 0

            else:
                best_level = get_best_level_for_downsample(64)
                if best_level > img_lvl:
                    current_vis_params['vis_level'] = img_lvl
                else:
                    current_vis_params[
                        'vis_level'] = best_level  # If slide is scanned at just one zoom level, set the segmetation
                    # level to zero, else to level 6
        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dimensions) == 1:
                current_seg_params['seg_level'] = 0

            else:
                best_level = get_best_level_for_downsample(64)
                if best_level > img_lvl:
                    current_seg_params['seg_level'] = img_lvl
                else:
                    current_seg_params['seg_level'] = best_level
        print('Image is scanned at level = {} and will be segmented at level = {} and visualised at level = {}'.format(
            img_lvl, current_seg_params['seg_level'], current_vis_params['vis_level']))
        keep_ids = str(current_seg_params['keep_ids'])
        if keep_ids != 'none' and len(keep_ids) > 0:
            str_ids = current_seg_params['keep_ids']
            current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['keep_ids'] = []

        exclude_ids = str(current_seg_params['exclude_ids'])
        if exclude_ids != 'none' and len(exclude_ids) > 0:
            str_ids = current_seg_params['exclude_ids']
            current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['exclude_ids'] = []
        # Calculate the dimension of the WSI at level we want to segment it and if it is too large abort the
        # segmentation
        dim = WSI_object.level_dimensions[int(current_seg_params['seg_level'])]
        w = dim[1] - dim[0] + 1
        h = dim[3] - dim[2] + 1
        if w * h > 1e8:
            print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)

        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id + '.jpg')
            mask.save(mask_path)

        patch_time_elapsed = -1  # Default time
        if patch:
            current_patch_params.update({
                                            'patch_level': patch_level, 'patch_size': patch_size,
                                            'step_size'  : step_size, 'save_path': patch_save_dir
                                        })
            file_path, patch_time_elapsed = patching(WSI_object=WSI_object, **current_patch_params, )

        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id + '.h5')
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id + '.jpg')
                heatmap.save(stitch_path)

        print("segmentation took {} seconds".format(seg_time_elapsed))
        print("patching took {} seconds".format(patch_time_elapsed))
        print("stitching took {} seconds".format(stitch_time_elapsed))

        # Update the slides information: staus, vis_level , seg_level, scanned_level
        df.loc[idx, 'status'] = 'processed'
        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']
        df.loc[idx, 'scanned_level'] = img_lvl

        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))

    return seg_times, patch_times


parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type=str, help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type=int, default=256, help='step_size')
parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type=str, help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
                    help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0, help='downsample level at which to patch')
parser.add_argument('--process_list', type=str, default=None,
                    help='name of list of images to process with parameters (.csv)')

if __name__ == '__main__':
    args = parser.parse_args()

    # The output directories to save masks, patches and stitches
    patch_save_dir = os.path.join(args.save_dir, 'patches')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')

    # If there is a list of files to be processed read it in
    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)
    else:
        process_list = None

    print('source: ', args.source)
    print('patch_save_dir: ', patch_save_dir)
    print('mask_save_dir: ', mask_save_dir)
    print('stitch_save_dir: ', stitch_save_dir)

    directories = {
        'source'       : args.source, 'save_dir': args.save_dir, 'patch_save_dir': patch_save_dir,
        'mask_save_dir': mask_save_dir, 'stitch_save_dir': stitch_save_dir
    }
    # Create the output directories if they do not exist
    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val,
                        exist_ok=True)  # Set the segmentation, filtering and visulization parameters and parameters
            # used for patching
    seg_params = {
        'seg_level'  : -1, 'sthresh': 10, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': 'none',
        'exclude_ids': 'none'
    }
    filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 20}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    # Read the .csv predefined profile of default segmentation and filter parameters if it exists
    if args.preset:
        preset_df = pd.read_csv(os.path.join('presets', args.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]

    parameters = {
        'seg_params': seg_params, 'filter_params': filter_params, 'patch_params': patch_params, 'vis_params': vis_params
    }

    print(parameters)

    # Do the segmentation, patching and Stitching if it is mentioned to do
    seg_times, patch_times = seg_and_patch(**directories, **parameters, patch_size=args.patch_size,
                                           step_size=args.step_size, seg=args.seg, use_default_params=False,
                                           save_mask=True, stitch=args.stitch, patch_level=args.patch_level,
                                           patch=args.patch, process_list=process_list, auto_skip=args.no_auto_skip)
