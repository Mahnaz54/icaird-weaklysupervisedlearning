import os
import cv2
import math
import time
import pdb
import copy
import h5py
import itertools

import numpy as np
import tempfile, uuid
import multiprocessing as mp
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from collections import defaultdict
from xml.dom import minidom
from PIL import Image
from matplotlib import cm
from wsi_core.wsi_utils import savePatchIter_bag_hdf5, initialize_hdf5_bag, coord_generator, save_hdf5, sample_indices, screen_coords, isBlackPatch, isWhitePatch, to_percentiles
from wsi_core.util_classes import isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, Contour_Checking_fn
from utils.file_utils import load_pkl, save_pkl

from libraries import pixelengine
import libraries.softwarerendercontext as softwarerendercontext
import libraries.softwarerenderbackend as softwarerenderbackend

render_context = softwarerendercontext.SoftwareRenderContext()
render_backend = softwarerenderbackend.SoftwareRenderBackend()
pe = pixelengine.PixelEngine(render_backend, render_context)

def get_best_level_for_downsample(downsample):
    best_level = int(math.log(downsample, 2))
    return best_level

def DrawGrid(img, coord, shape, thickness=250, color=(0,0,0,255)):
    cv2.rectangle(img, tuple(np.maximum([0, 0], coord-thickness//2)), tuple(coord - thickness//2 + np.array(shape)), (0, 0, 0, 255), thickness=thickness)
    return img


def DrawMap(canvas, patch_dset, coords, patch_size, indices=None, verbose=1, draw_grid=True):
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
        print('\nstart stitching {}.syntax ...'.format(patch_dset.attrs['wsi_name']))

    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(idx, total))

        patch_id = indices[idx]
        patch = patch_dset[patch_id]
        patch = cv2.resize(patch, patch_size)
        coord = coords[patch_id]
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)

def StitchPatches(h5py_file_path, downscale= 64, draw_grid=False, bg_color=(0,0,0), alpha=-1):
    file = h5py.File(h5py_file_path, 'r')
    dest = file['imgs']
    coords = file['coords'][:]

    if 'downsampled_level_dim' in dest.attrs.keys():
        w, h = dest.attrs['downsampled_level_dim']
    else:
        w, h = dset.attrs['level_dim']

    w = w // downscale
    h = h //downscale
    coords = (coords / downscale).astype(np.int32)
    #print('downscaled size for stiching: {} x {}'.format(w, h))
    print('number of patches: {}'.format(len(dest)))
    img_shape = dest[0].shape
    print('patch shape: {}'.format(img_shape))
    downscaled_shape = (img_shape[1] // downscale, img_shape[0] // downscale)

    if w*h > Image.MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)

    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w,h), mode = 'RGB', color = bg_color)
    else:
         heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))

    heatmap = np.array(heatmap)
    heatmap = DrawMap(heatmap, dest, coords, downscaled_shape, indices=None, draw_grid=draw_grid)

    file.close()
    return heatmap

def get_cache_folder():
    cache_path = os.path.join(tempfile.gettempdir(), 'cache')
    os.makedirs(cache_path, exist_ok=True)
    return cache_path


class RegionRequest:
    def __init__(self, loc, level, size):
        """

        Args:
            loc (int, int): position of the top left pixel of the region
            level (int): the level of the requested region
            size (int, int): size of the region to be read
        """
        self._loc = loc
        self._level = level
        self._size = size

    @property
    def loc(self):
        return self._loc

    @property
    def level(self):
        return self._level

    @property
    def size(self):
        return self._size

class WholeSlideImage(object):
    def __init__(self, path, hdf5_file=None):
        self.path = path
        self.pe = pe
        self.name = ".".join(path.split("/")[-1].split('.')[:-1])
        cache_folder = get_cache_folder()
        self.pe_slide_uuid = str(uuid.uuid4())
        self.pe[self.pe_slide_uuid].open(path, '', 'r', cache_folder)
        self._view = self.pe[self.pe_slide_uuid].SourceView()
        levels = self.pe[self.pe_slide_uuid].numLevels() + 1
        self.level_dimensions = [self._get_valid_range(l) for l in range(levels)]
              
        
        self.contours_tissue = None
        self.contours_tumor = None
        self.seg_level = None
        self.hdf5_file = hdf5_file
        
        
    def getOpenSlide(self):
        self.wsi = pe['in'].open(self.path)
        return self.wsi

    def __del__(self):
        if hasattr(self, 'pe'):
            self.pe[self.pe_slide_uuid].close()

    def _get_valid_range(self, level):
        """Get the valid range at the specified level.

        The valid range corresponds to the dimensionRanges of the given
        level clipped to the data envelopes (we assume that the regions
        between envelopes are also valid).
        """
        dim_ranges = self._view.dimensionRanges(level)

        start_x, end_x = dim_ranges[0][0], dim_ranges[0][2]
        start_y, end_y = dim_ranges[1][0], dim_ranges[1][2]

        envelope_polys = self._view.dataEnvelopes(0).asRectangles()
        min_x, min_y = min([e[0] for e in envelope_polys]), min([e[2] for e in envelope_polys])
        max_x, max_y = max([e[1] for e in envelope_polys]), max([e[3] for e in envelope_polys])
        return (int(max(min_x, start_x) / 2 ** level), int(min(max_x, end_x) / 2 ** level),
                int(max(min_y, start_y) / 2 ** level), int(min(max_y, end_y) / 2 ** level))

    def read_region(self, region_request):
        """Read a region from the slide at the specified level.

        Args:
            region_request (RegionRequest): the region to be requested
        Returns:
            numpy.core.ndarray: ndarray with the region extracted from the image
        """
        return next(self.read_regions([region_request]))[1]

    def read_regions(self, region_requests):
        """Read a part of the slide and returns as numpy array.

        Pixels not provided by the PixelEngine are padded with 255

        Args:
            region_requests (collections.Iterable[RegionRequest]): the regions to be requested
        Returns:
            collections.Generator[(RegionRequest, numpy.core.ndarray)]: with the region extracted from the image
        """
        view_ranges = defaultdict(list)
        valid_requests = defaultdict(list)
        for req in region_requests:
            x_min, y_min = req.loc
            width, height = req.size
            x_max = x_min + width - 1
            y_max = y_min + height - 1

            x_min *= 2 ** req.level
            x_max *= 2 ** req.level
            y_min *= 2 ** req.level
            y_max *= 2 ** req.level

            view_range = (x_min, x_max, y_min, y_max, req.level)
            view_ranges[req.level].append(view_range)
            valid_requests[req.level].append(req)

        regions_lut = dict()

        for level, vr in view_ranges.items():
            regions = self._view.requestRegions(vr,
                                                backgroundColor=[255, 255, 255, 255],
                                                enableAsyncRendering=True,
                                                dataEnvelopes=self._view.dataEnvelopes(level))
            regions_lut.update({region: req for req, region in zip(valid_requests[level], regions)})
            del regions


        del view_ranges
        del valid_requests

        while regions_lut:
            regions_ready = self.pe.waitAny(list(regions_lut.keys()))
            for region in regions_ready:
                req = regions_lut.pop(region)
                try:
                    size = (req.size[1], req.size[0], 3)
                    patch = np.empty(size, dtype=np.uint8)

                    region.get(patch)
                    yield req, patch
                except Exception as ex:
                    yield req, ex


    def initXML(self, xml_path):
        def _createContour(coord_list):
            return np.array([[[int(float(coord.attributes['X'].value)), 
                               int(float(coord.attributes['Y'].value))]] for coord in coord_list], dtype = 'int32')

        xmldoc = minidom.parse(xml_path)
        annotations = [anno.getElementsByTagName('Coordinate') for anno in xmldoc.getElementsByTagName('Annotation')]
        self.contours_tumor  = [_createContour(coord_list) for coord_list in annotations]
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def initTxt(self,annot_path):
        def _create_contours_from_dict(annot):
            all_cnts = []
            for idx, annot_group in enumerate(annot):
                contour_group = annot_group['coordinates']
                if annot_group['type'] == 'Polygon':
                    for idx, contour in enumerate(contour_group):
                        contour = np.array(contour).astype(np.int32).reshape(-1,1,2)
                        all_cnts.append(contour) 

                else:
                    for idx, sgmt_group in enumerate(contour_group):
                        contour = []
                        for sgmt in sgmt_group:
                            contour.extend(sgmt)
                        contour = np.array(contour).astype(np.int32).reshape(-1,1,2)    
                        all_cnts.append(contour) 

            return all_cnts
        with open(annot_path, "r") as f:
            annot = f.read()
            annot = eval(annot)
        self.contours_tumor  = _create_contours_from_dict(annot)
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def initSegmentation(self, mask_file):
        # load segmentation results from pickle file
        import pickle
        asset_dict = load_pkl(mask_file)
        self.holes_tissue = asset_dict['holes']
        self.contours_tissue = asset_dict['tissue']

    def saveSegmentation(self, mask_file):
        # save segmentation results using pickle
        asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}
        save_pkl(mask_file, asset_dict) 

    def segmentTissue(self, seg_level=0, sthresh=13, sthresh_up = 255, mthresh=7, close = 0, use_otsu=False, 
                            filter_params={'a':100}, ref_patch_size=32, exclude_ids=[], keep_ids=[]):
        """
            Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """
        
        def _filter_contours(contours, hierarchy, filter_params):
            """
                Filter contours by: area.
            """
            filtered = []

            # find indices of foreground contours (parent == -1)
            hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
            all_holes = []
            #loop through froeground contour indices
            for cont_idx in hierarchy_1:
                cont = contours[cont_idx]
                # indices of holes contained in this contour (children of parent contour)
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                # take contour area (includes holes)
                a = cv2.contourArea(cont)
                # calculate the contour area of each hole
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                # actual area of foreground contour region
                a = a - np.array(hole_areas).sum()
                if a == 0: continue
                if tuple((filter_params['a_t'],)) < tuple((a,)): 
                    filtered.append(cont_idx)
                    all_holes.append(holes)

            foreground_contours = [contours[cont_idx] for cont_idx in filtered]
            hole_contours = []

            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids ]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = []
                
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)
                hole_contours.append(filtered_holes)
            return foreground_contours, hole_contours
        seg_level = int(seg_level)
             
        #get size of image at the segmentation level given
        dim = self.level_dimensions[seg_level]
        w = dim[1]-dim[0]+1
        h = dim[3]-dim[2]+1
        loc =(dim[0], dim[2])
        patch_request = RegionRequest(loc, seg_level, (w,h))
        
        img = np.array(self.read_region(patch_request))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # Apply median blurring
        
       
        # Thresholding
        if use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)                 
        scale = tuple(ele1 // ele2 for ele1, ele2 in zip(self.level_dimensions[0], (loc[0], w, loc[1], h)))
        scale =(scale[1], scale[3])
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area

        # Find and filter contours
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:,2:]
        
        if filter_params: foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts
        self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
        self.holes_tissue = self.scaleHolesDim(hole_contours, scale)
        self.seg_level = seg_level
        
        if len(keep_ids) > 0:
            contour_ids = set(keep_ids) - set(exclude_ids)
        else:
            contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)

        self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
        self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]
    # hole_color = blue
    # color = green
    # annot_color = red

    def visWSI(self, vis_level=0, color = (0,255,0), hole_color = (0,0,255), annot_color=(255,0,0), 
                    line_thickness=12, max_size=None, top_left=None, bot_right=None, custom_downsample=1, view_slide_only=False,
                    number_contours=False, seg_display=True, annot_display=True):  
        
        lvl = int(vis_level)
        downsample = tuple(ele1 // ele2 for ele1, ele2 in zip(self.level_dimensions[0], self.level_dimensions[lvl]))
        downsample = (downsample[1], downsample[3])

        scale = [1/downsample[0], 1/downsample[1]] 
        
        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            top_left = (0, 0)
            dim = self.level_dimensions[lvl]
            region_size = (dim[1]-dim[0]+1, dim[3]-dim[2]+1)

        patch_request = RegionRequest(top_left, lvl, region_size)
        img = np.array(self.read_region(patch_request))
        
        if not view_slide_only:
            offset = tuple(-(np.array(top_left) * scale).astype(int))
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            if self.contours_tissue is not None and seg_display:
                if not number_contours:
                    cv2.drawContours(img, self.scaleContourDim(self.contours_tissue, scale),
                                     -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)

                else: # add numbering to each contour
                    for idx, cont in enumerate(self.contours_tissue):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        M = cv2.moments(contour)
                        cX = int(M["m10"] / (M["m00"] + 1e-9))
                        cY = int(M["m01"] / (M["m00"] + 1e-9))
                        # draw the contour and put text next to center
                        cv2.drawContours(np.array(img),  [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                        cv2.putText(np.array(img), "{}".format(idx), (cX, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)

                for holes in self.holes_tissue:
                    cv2.drawContours(np.array(img), self.scaleContourDim(holes, scale),
                                     -1, hole_color, line_thickness, lineType=cv2.LINE_8)

            if self.contours_tumor is not None and annot_display:
                cv2.drawContours(np.array(img), self.scaleContourDim(self.contours_tumor, scale),
                                 -1, annot_color, line_thickness, lineType=cv2.LINE_8, offset=offset)

        
        img = Image.fromarray(img)
        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
       
        return img


    def createPatches_bag_hdf5(self, save_path, patch_level=0, patch_size=256, step_size=256, save_coord=True, **kwargs):
        contours = self.contours_tissue
        contour_holes = self.holes_tissue

        print('\nCreating patches for {}.isyntax ...\n'.format(self.name))
        elapsed = time.time()
        for idx, cont in enumerate(contours):
            patch_gen = self._getPatchGenerator(cont, idx, patch_level, save_path, patch_size, step_size, **kwargs)
            
            if self.hdf5_file is None:
                try:
                    first_patch = next(patch_gen)

                # empty contour, continue
                except StopIteration:
                    continue

                file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
                self.hdf5_file = file_path

            for patch in patch_gen:
                savePatchIter_bag_hdf5(patch)

        return self.hdf5_file


    def _getPatchGenerator(self, cont, cont_idx, patch_level, save_path, patch_size=256, step_size=256, custom_downsample=1,
        white_black=True, white_thresh=15, black_thresh=50, contour_fn='four_pt', use_padding=True): 
        lvl = patch_level
        
        w = self.level_dimensions[lvl][1] - self.level_dimensions[lvl][0] + 1  #
        h = self.level_dimensions[lvl][3] - self.level_dimensions[lvl][2] + 1  #
        
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, w,h)
        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))
        if custom_downsample > 1:
            assert custom_downsample == 2 
            # the target size is what's specified by patch_size
            target_patch_size = patch_size 
            # the actual patches that we want to take is 2 * target_size for each dimension
            patch_size = target_patch_size * 2 
            # similarly, the step size is 2 * what's specified
            step_size = step_size * 2
            print("Custom Downsample: {}, Patching at {} x {}, But Final Patch Size is {} x {}".format(custom_downsample, patch_size, patch_size, 
                target_patch_size, target_patch_size))
        
        if patch_level == 0:
            patch_downsample =(1, 1) 
        else:
            downsample = tuple(ele1 // ele2 for ele1, ele2 in zip(self.level_dimensions[0], self.level_dimensions[patch_level]))
            patch_downsample = (downsample[1], downsample[3])

        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])
        # step sizes to take at levl 0
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]
        
        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        #image size at level zero
        img_w = self.level_dimensions[0][1] - self.level_dimensions[0][0] + 1
        img_h = self.level_dimensions[0][3] - self.level_dimensions[0][2] + 1

        if use_padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            stop_y = min(start_y+h, img_h-ref_patch_size[1])
            stop_x = min(start_x+w, img_w-ref_patch_size[0])

        count = 0
        for y in range(start_y, stop_y, step_size_y):
            for x in range(start_x, stop_x, step_size_x):

                if not self.isInContours(cont_check_fn, cont, (x,y), self.holes_tissue[cont_idx], ref_patch_size[0]): #point not inside contour and its associated holes
                    continue    
                
                count+=1
                patch_request = RegionRequest((x,y), patch_level, (patch_size, patch_size))
                img = np.array(self.read_region(patch_request))
                patch_PIL = Image.fromarray(img.astype(np.uint8))

                if custom_downsample > 1:
                    patch_PIL = patch_PIL.resize((target_patch_size, target_patch_size))
                
                if white_black:
                    if self.isBlackPatch(np.array(patch_PIL), rgbThresh=black_thresh) or self.isWhitePatch(np.array(patch_PIL), satThresh=white_thresh): 
                        continue

                downsampled_dim = tuple(np.array(self.level_dimensions[patch_level])//custom_downsample)
                downsample_level_dim = tuple((downsampled_dim[1], downsampled_dim[3]))
                
                level_dim = self.level_dimensions[patch_level]
                level_dim = tuple((level_dim[1], level_dim[3]))
                patch_info = {'x':x // (patch_downsample[0] * custom_downsample), 'y':y // (patch_downsample[1] * custom_downsample), 'cont_idx':cont_idx, 'patch_level':patch_level,
                'downsample': patch_downsample, 'downsampled_level_dim': downsample_level_dim , 'level_dim': level_dim,
                'patch_PIL':patch_PIL, 'name':self.name, 'save_path':save_path}

                yield patch_info
        print("patches extracted: {}\n".format(count))


    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
                return 1
        
        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        if cont_check_fn(pt):
            if holes is not None:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0


    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]
   
    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]

    def process_contours(self, save_path, patch_level=0, patch_size=256, step_size=256, **kwargs):
        save_path_hdf5 = os.path.join(save_path, str(self.name) + '.h5')
        print("Creating patches for: ", self.name, "...",)
        elapsed = time.time()
        n_contours = len(self.contours_tissue)
        print("Total number of contours to process: ", n_contours)
        fp_chunk_size = math.ceil(n_contours * 0.05)
        init = True
        for idx, cont in enumerate(self.contours_tissue):
            if (idx + 1) % fp_chunk_size == fp_chunk_size:
                print('Processing contour {}/{}'.format(idx, n_contours))
            
            asset_dict, attr_dict = self.process_contour(cont, self.holes_tissue[idx], patch_level, save_path, patch_size, step_size, **kwargs)
            if len(asset_dict) > 0:
                if init:
                    save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                    init = False
                else:
                    save_hdf5(save_path_hdf5, asset_dict, mode='a')

        return self.hdf5_file


    def process_contour(self, cont, contour_holes, patch_level, save_path, patch_size = 256, step_size = 256,
        contour_fn='four_pt', use_padding=True, top_left=None, bot_right=None):
        #get size of images at the level patching is done
        w = self.level_dimensions[patch_level][1] - self.level_dimensions[patch_level][0] + 1 
        h = self.level_dimensions[patch_level][3] - self.level_dimensions[patch_level][2] + 1 
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, w, h)
        
        #if patching needs to be done at any level other than zero the patch_size needs to be downsampled.
        if patch_level == 0:
            patch_downsample = (1,1)
        else:
            level_downsample = tuple(ele1 // ele2 for ele1, ele2 in zip(self.level_dimensions[0], self.level_dimensions[patch_level]))
            patch_downsample = (level_downsample[1],level_downsample[3])
        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])
        
        #size of image at level zero
        dim_0 = self.level_dimensions[0]
        img_w, img_h = dim_0[1]-dim_0[0]+1, dim_0[3]-dim_0[2]+1

        if use_padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            stop_y = min(start_y+h, img_h-ref_patch_size[1]+1)
            stop_x = min(start_x+w, img_w-ref_patch_size[0]+1)
        
        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))
        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        if bot_right is not None or top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            if w <= 0 or h <= 0:
                print("Contour is not in specified ROI, skip")
                return {}, {}
            else:
                print("Adjusted Bounding Box:", start_x, start_y, w, h)
        
        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn
        
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]
        x_range = np.arange(start_x, stop_x, step=step_size_x)
        y_range = np.arange(start_y, stop_y, step=step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

        if len(coord_candidates) < mp.cpu_count():
            num_workers = len(coord_candidates)
        else:
            num_workers = mp.cpu_count()
       
        pool = mp.Pool(num_workers)
        iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
        results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
        pool.close()
        results = np.array([result for result in results if result is not None])
        downsampled_level_dim = tuple((img_w //patch_downsample[0] , img_h//patch_downsample[1]))
        level_dim = self.level_dimensions[patch_level]
        level_dim = tuple((level_dim[1], level_dim[3]))
        

        print('Extracted {} coordinates'.format(len(results)))

        if len(results)>1:
            asset_dict = {'coords' :          results}
            
            attr = {'patch_size' :            patch_size, # To be considered...
                    'patch_level' :           patch_level,
                    'downsample':             patch_downsample, 
                    'downsampled_level_dim' : downsampled_level_dim,
                    'level_dim':              level_dim,
                    'name':                   self.name,
                    'save_path':              save_path}

            attr_dict = { 'coords' : attr}
            return asset_dict, attr_dict

        else:
            return {}, {}

    @staticmethod
    def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn):
        if WholeSlideImage.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
            return coord
        else:
            return None

    def visHeatmap(self, scores, coords, vis_level=-1, 
                   top_left=None, bot_right=None,
                   patch_size=(256, 256), 
                   blank_canvas=False, canvas_color=(220, 20, 50), alpha=0.4, 
                   blur=False, overlap=0.0, 
                   segment=True, use_holes=True,
                   convert_to_percentiles=False, 
                   binarize=False, thresh=0.5,
                   max_size=None,
                   custom_downsample = 1,
                   cmap='coolwarm'):

        """
        Args:
            scores (numpy array of float): Attention scores 
            coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
            vis_level (int): WSI pyramid level to visualize
            patch_size (tuple of int): Patch dimensions (relative to lvl 0)
            blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
            canvas_color (tuple of uint8): Canvas color
            alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
            blur (bool): apply gaussian blurring
            overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
            segment (bool): whether to use tissue segmentation contour (must have already called self.segmentTissue such that 
                            self.contours_tissue and self.holes_tissue are not None
            use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
            convert_to_percentiles (bool): whether to convert attention scores to percentiles
            binarize (bool): only display patches > threshold
            threshold (float): binarization threshold
            max_size (int): Maximum canvas size (clip if goes over)
            custom_downsample (int): additionally downscale the heatmap by specified factor
            cmap (str): name of matplotlib colormap to use
        """
        vis_level = int(vis_level)
        if vis_level < 0:
            vis_level = get_best_level_for_downsample(32)

        elif vis_level == 0:
            downsample = (1,1)
        else:
            downsample = tuple(ele1 // ele2 for ele1, ele2 in zip(self.level_dimensions[0], self.level_dimensions[vis_level]))
            downsample = (downsample[1], downsample[3] )
        scale = [1/downsample[0], 1/downsample[1]] # Scaling from 0 to desired level
        if len(scores.shape) == 2:
            scores = scores.flatten()
        if binarize:
            if thresh < 0:
                threshold = 1.0/len(scores)
            else:
                threshold =  thresh
        
        else:
            threshold = 0.0
        ##### calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
        if top_left is not None and bot_right is not None:
            scores, coords = screen_coords(scores, coords, top_left, bot_right)
            coords = coords - top_left
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            region_dim = self.level_dimensions[int(vis_level)]
            region_size = region_dim[1]-region_dim[0]+1, region_dim[3]-region_dim[2]+1
            top_left = (0,0)
            dim = self.level_dimensions[0]
            bot_right = tuple((dim[1]-dim[0]+1, dim[3]-dim[2]+1))
            
            w, h = region_size
        
        patch_size  = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)
        
        print('\ncreating heatmap for: ')
        print('top_left: ', top_left, 'bot_right: ', bot_right)
        print('w: {}, h: {}'.format(w, h))
        print('scaled patch size: ', patch_size)
        ###### normalize filtered scores ######
        if convert_to_percentiles:
            scores = to_percentiles(scores) 
        scores /= 100
        ######## calculate the heatmap of raw attention scores (before colormap) 
        # by accumulating scores over overlapped regions ######
        
        # heatmap overlay: tracks attention score over each pixel of heatmap
        # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
        overlay = np.full(np.flip(region_size), 0).astype(float)
        counter = np.full(np.flip(region_size), 0).astype(np.uint16)      
        count = 0
        for idx in range(len(coords)):
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:
                if binarize:
                    score=1.0
                    count+=1
            else:
                score=0.0
            # accumulate attention
            overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += score
            # accumulate counter
            counter[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += 1

        if binarize:
            print('\nbinarized tiles based on cutoff of {}'.format(threshold))
            print('identified {}/{} patches as positive'.format(count, len(coords)))
        
        # fetch attended region and average accumulated attention
        zero_mask = counter == 0
        if binarize:
            overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
        else:
            overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
        del counter 
        if blur:
            overlay = cv2.GaussianBlur(overlay,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  
        if segment:
            tissue_mask = self.get_seg_mask(region_size, scale, use_holes=use_holes, offset=tuple(top_left))
            # return Image.fromarray(tissue_mask) # tissue mask
        if not blank_canvas:
            # downsample original image and use as canvas
            region_request = RegionRequest(top_left,vis_level, region_size)
            img= np.array(self.read_region(region_request))
            #img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
        else:
            # use blank canvas
            img = np.array(Image.new(size=region_size, mode="RGB", color=(255,255,255))) 

        #return Image.fromarray(img) #raw image

        print('\ncomputing heatmap image')
        print('total of {} patches'.format(len(coords)))
        twenty_percent_chunk = max(1, int(len(coords) * 0.2))

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        
        for idx in range(len(coords)):
            if (idx + 1) % twenty_percent_chunk == 0:
                print('progress: {}/{}'.format(idx, len(coords)))
            
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:

                # attention block
                raw_block = overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]]
                
                # image block (either blank canvas or orig image)
                img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()

                # color block (cmap applied to attention block)
                color_block = (cmap(raw_block) * 255)[:,:,:3].astype(np.uint8)

                if segment:
                    # tissue mask block
                    mask_block = tissue_mask[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] 
                    # copy over only tissue masked portion of color block
                    img_block[mask_block] = color_block[mask_block]
                else:
                    # copy over entire color block
                    img_block = color_block

                # rewrite image block
                img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = img_block.copy()
        
        #return Image.fromarray(img) #overlay
        print('Done')
        del overlay
        if blur:
            img = cv2.GaussianBlur(img,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  
        if alpha < 1.0:
            img = self.block_blending(img, vis_level, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas, block_size=1024)
        
        img = Image.fromarray(img)
        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
        return img

    
    def block_blending(self, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
        print('\ncomputing blend')
        downsample = tuple(ele1 // ele2 for ele1, ele2 in zip(self.level_dimensions[0], self.level_dimensions[vis_level]))
        downsample = (downsample[1], downsample[3])
        w = img.shape[1]
        h = img.shape[0]
        block_size_x = min(block_size, w)
        block_size_y = min(block_size, h)
        print('using block size: {} x {}'.format(block_size_x, block_size_y))

        shift = top_left # amount shifted w.r.t. (0,0)
        for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
            for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
        
                # 1. convert wsi coordinates to image coordinates via shift and scale
                x_start_img = int((x_start - shift[0]) / int(downsample[0]))
                y_start_img = int((y_start - shift[1]) / int(downsample[1]))
                
                # 2. compute end points of blend tile, careful not to go over the edge of the image
                y_end_img = min(h, y_start_img+block_size_y)
                x_end_img = min(w, x_start_img+block_size_x)

                if y_end_img == y_start_img or x_end_img == x_start_img:
                    continue
                      
                # 3. fetch blend block and size
                blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img] 
                blend_block_size = (x_end_img-x_start_img, y_end_img-y_start_img)
                
                if not blank_canvas:
                    # 4. read actual wsi block as canvas block
                    pt = (x_start, y_start)
                    region_request = RegionRequest(pt, vis_level, blend_block_size)
                    canvas = np.array(self.read_region(region_request))
                    #canvas = Image.fromarray(canvas.astype(np.uint8))
                else:
                    # 4. OR create blank canvas block
                    canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255,255,255)))

                # 5. blend color block and canvas block
                img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas, 1 - alpha, 0, canvas)
        return img

    def get_seg_mask(self, region_size, scale, use_holes=True, offset=(0,0)):
        print('\ncomputing foreground tissue mask')
        tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)
        contours_tissue = self.scaleContourDim(self.contours_tissue, scale)
        offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))
        contours_holes = self.scaleHolesDim(self.holes_tissue, scale)
        contours_tissue, contours_holes = zip(*sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True))
        for idx in range(len(contours_tissue)):
            cv2.drawContours(image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1), offset=offset, thickness=-1)
            
            if use_holes:
                cv2.drawContours(image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0), offset=offset, thickness=-1)
        tissue_mask = tissue_mask.astype(bool)
        print('detected {}/{} of region as tissue'.format(tissue_mask.sum(), tissue_mask.size))
        return tissue_mask


