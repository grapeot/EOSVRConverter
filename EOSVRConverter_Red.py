"""
This script is specifically designed for the RED V-Raptor VV, which is compatible with the Canon RF mount and can be sued with the US VR lens. The differences between this script and the original Canon one are as follows: 

1. The camera parameters are specifically calibrated for the V-Raptor VV. 
2. This version employs Ray for distributed computing, which offers improved robustness compared to the original multiprocessing-based method. 
3. The workflow is changed. We expect that color grading will be conducted in DaVinci, not FFmpeg. We've also removed the shadow and highlight adjustments for the same reason.
4. We expect the script's output to be further processed, typically in Topaz Sharpener AI and DaVinci for the final sharpening, rendering and encoding. As a result, we've eliminated the FFmpeg encoding section. 
"""

import numpy as np
import cv2
from subprocess import Popen
from random import randint
from subprocess import Popen
from tqdm import tqdm
from os import getcwd, listdir, cpu_count, mkdir
from os.path import exists, join
from glob import glob
from time import sleep
from Rotator3D import Rotator3D
from shutil import move
import logging
import ray
from ray.util.queue import Queue

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
TOPAZ_BIN = r"C:\Program Files\Topaz Labs LLC\Topaz Sharpen AI\Topaz Sharpen AI.exe"
FILE_EXTNAME = '.tif'

# From https://github.com/kylemcdonald/FisheyeToEquirectangular
class FisheyeToEquirectangular:
    FISHEYE_FILENAME = join(getcwd(), 'fisheye_red.npy')

    def __init__(self, n=4096, side=3200, blending=0, aperture=1):
        """
        :param shadow_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
        :param shadow_tone_percent [0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
        :param shadow_radius [>0]: Controls the size of the local neighborhood around each pixel
        :param highlight_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
        :param highlight_tone_percent [0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
        :param highlight_radius [>0]: Controls the size of the local neighborhood around each pixel
        :param color_percent [-1.0 ~ 1.0]:
        """
        self.blending = blending
        blending_ratio = blending / n
        if exists(FisheyeToEquirectangular.FISHEYE_FILENAME):
            data = np.load(FisheyeToEquirectangular.FISHEYE_FILENAME)
            self.x, self.y = data
            # print(f'Loaded data: {data}')
        else:
            x_samples = np.linspace(0-blending_ratio, 1+blending_ratio, n+blending*2)
            y_samples = np.linspace(-1, 1, n)

            # equirectangular
            x, y = np.meshgrid(x_samples, y_samples)

            # longitude/latitude
            longitude = x * np.pi
            latitude = y * np.pi / 2

            # 3d vector
            Px = np.cos(latitude) * np.cos(longitude)
            Py = np.cos(latitude) * np.sin(longitude)
            Pz = np.sin(latitude)

            # 2d fisheye
            aperture *= np.pi
            r = 2 * np.arctan2(np.sqrt(Px*Px + Pz*Pz), Py) / aperture
            theta = np.arctan2(Pz, Px)
            theta += np.pi
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            x = np.clip(x, -1, 1)
            y = np.clip(y, -1, 1)

            x = (-x + 1) * side / 2
            y = (y + 1) * side / 2

            self.x = x.astype(np.float32)
            self.y = y.astype(np.float32)
            data = [self.x, self.y]
            np.save(FisheyeToEquirectangular.FISHEYE_FILENAME, data)

    def unwarp_single(self, img, interpolation=cv2.INTER_LINEAR, border=cv2.BORDER_REFLECT):
        return cv2.remap(
            img, self.x, self.y,
            interpolation=interpolation,
            borderMode=border
        )

    def getLeftRightFisheyeImage(self, img):
        h, w, c = img.shape
        # Use the intrinsic matrix from actual calibration
        centerLx = 2248
        centerRx = 5967
        centerY = 2250
        fisheyeR = 1600
        imgL = img[centerY - fisheyeR: centerY + fisheyeR, centerLx - fisheyeR: centerLx + fisheyeR, :]
        imgR = img[centerY - fisheyeR: centerY + fisheyeR, centerRx - fisheyeR: centerRx + fisheyeR, :]
        return imgL, imgR

    def correctForImage(self, img, outfn, rotator=None):
        if type(img) == str:
            imgpath = img
            try:
                img = cv2.imread(img)
            except:
                print(f'Failed to load image {imgpath}')
                return None
        if img is None:
            print(f'Failed to load image {imgpath}')
            return None
        imgL, imgR = self.getLeftRightFisheyeImage(img)
        newimg = self.unwarp_single(imgL)
        newimgR = cv2.rotate(newimg, cv2.ROTATE_180)
        if rotator is None:
            newimg = self.unwarp_single(imgR)
            newimgL = cv2.rotate(newimg, cv2.ROTATE_180)
            newimg = np.hstack((newimgL, newimgR))
        else:
            # For the rotating case, we only need one eye
            newimg = rotator.transformOneImage(newimgR)
        cv2.imwrite(outfn, newimg)

    # Correct all images under the correct directory in place
    def correctAllImages(self, pool):
        fns = glob('*.jpg')
        pool.starmap(self.correctForImage, tqdm([(fn, fn) for fn in fns]))
        # Use Topaz Sharpen AI to enhance resolution
        command = [TOPAZ_BIN] + fns
        process = Popen(command)
        process.wait()

@ray.remote(num_cpus=1, memory=3*1024*1024*1024)
def launchWarpTask(correctorRef, fn):
    img = cv2.imread(fn, -1)
    return correctorRef.correctForImage(img, fn, None)

def correctForVideo(outdir):
    if not exists(outdir):
        mkdir(outdir)
    
    # Perform the mapping in parallel
    fns = glob(f'{outdir}/*{FILE_EXTNAME}')
    fns = [join(getcwd(), fn) for fn in fns]
    corrector = FisheyeToEquirectangular()
    correctorRef = ray.put(corrector)

    tasks = []
    for fn in tqdm(fns, desc='submitting jobs...'):
        tasks.append(launchWarpTask.remote(correctorRef, fn))
    tasks = [x for x in tasks if x is not None]
    for t in tqdm(tasks):
        ray.get(t)

if __name__ == '__main__':
    ray.init()
    correctForVideo('RedTest')
