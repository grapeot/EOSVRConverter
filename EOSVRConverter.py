import numpy as np
import cv2
from subprocess import Popen
from multiprocessing import Pool
from tqdm import tqdm
from os import mkdir, listdir
from os.path import exists, join

# From https://github.com/kylemcdonald/FisheyeToEquirectangular
class FisheyeToEquirectangular:
    def __init__(self, n=4096, side=3600, blending=16, aperture=1):
        self.blending = blending
        blending_ratio = blending / n
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
    
    def unwarp_single(self, img, interpolation=cv2.INTER_LINEAR, border=cv2.BORDER_REFLECT):
        return cv2.remap(
            img, self.x, self.y,
            interpolation=interpolation,
            borderMode=border
        )

    def getLeftRightFisheyeImage(self, imgfn):
        img = cv2.imread(imgfn)
        h, w, c = img.shape
        centerLx = w // 4 - 50
        centerRx = w * 3 // 4 + 50
        centerY = h // 2 + 50
        fisheyeR = 1800
        imgL = img[centerY - fisheyeR: centerY + fisheyeR, centerLx - fisheyeR: centerLx + fisheyeR, :]
        imgR = img[centerY - fisheyeR: centerY + fisheyeR, centerRx - fisheyeR: centerRx + fisheyeR, :]
        return imgL, imgR

    def correctForImage(self, imgfn, outfn):
        imgL, imgR = self.getLeftRightFisheyeImage(imgfn)
        newimg = self.unwarp_single(imgL)
        newimgR = cv2.rotate(newimg, cv2.ROTATE_180)
        newimg = self.unwarp_single(imgR)
        newimgL = cv2.rotate(newimg, cv2.ROTATE_180)
        newimg = np.hstack((newimgL, newimgR))
        cv2.imwrite(outfn, newimg)

    # Extract frames from the video using ffmpeg, and then perform correction for each frame (in place)
    # Note the video here could be exported from Premiere or other software, and not necessarily the
    # out-of-body mp4 files. So even RAW could be supported (indirectly).
    def correctForVideo(self, videofn, outdir):
        if not exists(outdir):
            mkdir(outdir)
        ffmpegCommand = ['ffmpeg', '-i', videofn, '-qscale:v', '2', join(outdir, "%5d.png")]
        exe = Popen(ffmpegCommand)
        exe.wait()
        fns = [join(outdir, x) for x in listdir(outdir)]
        pool = Pool(120)
        pool.starmap(self.correctForImage, tqdm([(fn, fn) for fn in fns]))


if __name__ == '__main__':
    # We don't have a command line interface for now to provide maximum 
    # efficiency (e.g. no need to intialize FisheyeToEquirectangular every time)
    converter = FisheyeToEquirectangular()
    converter.correctForImage('HorizontalTest2.jpg', 'HT2.jpg')
    converter.correctForVideo('IMG_3784.mp4', 'frames')
