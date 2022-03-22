import os
import h5py
import cv2 as cv
import argparse

import numpy as np
import matplotlib.pyplot as plt
from wsi_core.WholeSlideImage import WholeSlideImage


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


parser = argparse.ArgumentParser(description='Patch Generation')
parser.add_argument('--slide_dir', type = str,
                                        help = 'path to slides')
parser.add_argument('--source', type = str,
                                        help='path to patch files')
parser.add_argument('--save_dir', type = str,
                                        help='path to store generated patches')

if __name__ == '__main__':
        args = parser.parse_args()

        patch_size = 256
        seg_level = 6

        patch_save_dir = os.path.join(args.save_dir, 'generated_patches')

        print('patch coordinate file path : ', args.source)
        print('patch_save_dir: ', patch_save_dir)


        for filename in sorted(os.listdir(args.slide_dir)):
            print('Extracting patches from {} .............'.format(filename))
            slide_path = os.path.join(args.slide_dir, filename)
            slidename = filename.split('.')[0]
            os.makedirs(os.path.join(patch_save_dir, slidename), exist_ok = True)
            h5_filename = slidename+'.h5'
            with h5py.File(os.path.join(args.source, h5_filename), 'r') as f:
                # Get the data
                coords = f['coords']
                wsi = WholeSlideImage(slide_path, hdf5_file = None)
                for i in range(len(coords)):
                    coord = coords[i]
                    patch_level = f['coords'].attrs['patch_level']
                    patch_size = f['coords'].attrs['patch_size']
                    region_request = RegionRequest(coord, patch_level, (patch_size,patch_size))
                    img = np.array(wsi.read_region(region_request))
                    patch_path = os.path.join(patch_save_dir, slidename)
                    patch_name =str(slidename) + '_'+ str(i) +'.jpg'
                    #plt.savefig(patch_path + patch_name, img)
                    cv.imwrite(os.path.join(patch_path,patch_name), img)