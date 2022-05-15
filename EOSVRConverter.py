import numpy as np
import cv2
from subprocess import Popen
from multiprocessing import Pool
from subprocess import Popen
from tqdm import tqdm
from os import mkdir, listdir
from os.path import exists, join
from glob import glob
import pickle

TOPAZ_BIN = r"C:\Program Files\Topaz Labs LLC\Topaz Sharpen AI\Topaz Sharpen AI.exe"

# Code adapted from https://gist.github.com/HViktorTsoi/8e8b0468a9fb07842669aa368382a7df
# Did some changes to speed up by 33% while nearly not changed the result
def shadowHighlightSaturationAdjustment(
        img,
        shadow_amount_percent, shadow_tone_percent, shadow_radius,
        highlight_amount_percent, highlight_tone_percent, highlight_radius,
        color_percent
):
    """
    Image Shadow / Highlight Correction. The same function as it in Photoshop / GIMP
    :param img: input RGB image numpy array of shape (height, width, 3)
    :param shadow_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
    :param shadow_tone_percent [0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
    :param shadow_radius [>0]: Controls the size of the local neighborhood around each pixel
    :param highlight_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
    :param highlight_tone_percent [0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
    :param highlight_radius [>0]: Controls the size of the local neighborhood around each pixel
    :param color_percent [-1.0 ~ 1.0]:
    :return:
    """
    shadow_tone = shadow_tone_percent * 255
    highlight_tone = 255 - highlight_tone_percent * 255

    shadow_gain = 1 + shadow_amount_percent * 6
    highlight_gain = 1 + highlight_amount_percent * 6

    # The entire correction process is carried out in YUV space,
    # adjust highlights/shadows in Y space, and adjust colors in UV space
    # convert to Y channel (grey intensity) and UV channel (color)
    height, width = img.shape[:2]
    imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV).astype('float32')
    img_Y, img_U, img_V = imgYUV[..., 0].reshape(-1), imgYUV[..., 1].reshape(-1), imgYUV[..., 2].reshape(-1)
    img_U -= 127
    img_V -= 127

    # extract shadow / highlight
    shadow_map = 255 - img_Y * 255 / shadow_tone
    shadow_map[np.where(img_Y >= shadow_tone)] = 0
    highlight_map = 255 - (255 - img_Y) * 255 / (255 - highlight_tone)
    highlight_map[np.where(img_Y <= highlight_tone)] = 0

    # // Gaussian blur on tone map, for smoother transition
    if shadow_amount_percent * shadow_radius > 0:
        # shadow_map = cv2.GaussianBlur(shadow_map.reshape(height, width), ksize=(shadow_radius, shadow_radius), sigmaX=0).reshape(-1)
        shadow_map = cv2.blur(shadow_map.reshape(height, width), ksize=(shadow_radius, shadow_radius)).reshape(-1)

    if highlight_amount_percent * highlight_radius > 0:
        # highlight_map = cv2.GaussianBlur(highlight_map.reshape(height, width), ksize=(highlight_radius, highlight_radius), sigmaX=0).reshape(-1)
        highlight_map = cv2.blur(highlight_map.reshape(height, width), ksize=(highlight_radius, highlight_radius)).reshape(-1)

    # Tone LUT
    t = np.arange(256)
    LUT_shadow = (1 - np.power(1 - t * (1 / 255), shadow_gain)) * 255
    LUT_shadow = np.maximum(0, np.minimum(255, np.int_(LUT_shadow + .5)))
    LUT_highlight = np.power(t * (1 / 255), highlight_gain) * 255
    LUT_highlight = np.maximum(0, np.minimum(255, np.int_(LUT_highlight + .5)))

    # adjust tone
    shadow_map = shadow_map * (1 / 255)
    highlight_map = highlight_map * (1 / 255)

    iH = (1 - shadow_map) * img_Y + shadow_map * LUT_shadow[np.int_(img_Y)]
    iH = (1 - highlight_map) * iH + highlight_map * LUT_highlight[np.int_(iH)]
    img_Y = iH

    # adjust color
    if color_percent != 0:
        # color LUT
        if color_percent > 0:
            LUT = (1 - np.sqrt(np.arange(32768)) * (1 / 128)) * color_percent + 1
        else:
            LUT = np.sqrt(np.arange(32768)) * (1 / 128) * color_percent + 1

        # adjust color saturation adaptively according to highlights/shadows
        color_gain = LUT[np.int_(img_U ** 2 + img_V ** 2 + .5)]
        w = 1 - np.minimum(2 - (shadow_map + highlight_map), 1)
        img_U = w * img_U + (1 - w) * img_U * color_gain
        img_V = w * img_V + (1 - w) * img_V * color_gain

    # re convert to RGB channel
    img_Y = img_Y.astype('uint8')
    img_U = (img_U + 127).astype('uint8')
    img_V = (img_V + 127).astype('uint8')
    imgYUV = np.row_stack([img_Y, img_U, img_V]).T.reshape(height, width, 3)
    output = cv2.cvtColor(imgYUV, cv2.COLOR_YUV2BGR)
    return output


# From https://github.com/kylemcdonald/FisheyeToEquirectangular
class FisheyeToEquirectangular:
    def __init__(self, n=4096, side=3600, blending=0, aperture=1):
        self.blending = blending
        blending_ratio = blending / n
        if exists('fisheye.npy'):
            data = np.load('fisheye.npy')
            self.x, self.y = data
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
            np.save('fisheye.npy', data)
    
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

    def correctForImage(self, imgfn, outfn, shallAdjust=False):
        imgL, imgR = self.getLeftRightFisheyeImage(imgfn)
        newimg = self.unwarp_single(imgL)
        newimgR = cv2.rotate(newimg, cv2.ROTATE_180)
        newimg = self.unwarp_single(imgR)
        newimgL = cv2.rotate(newimg, cv2.ROTATE_180)
        newimg = np.hstack((newimgL, newimgR))
        if shallAdjust:
            # Perform some image processing
            newimg = shadowHighlightSaturationAdjustment(newimg, 0.05, 0.4, 50, 0.1, 0.4, 50, 0.4)
        cv2.imwrite(outfn, newimg)

    # Correct all images under the correct directory in place
    def correctAllImages(self, pool):
        fns = glob('*.jpg')
        pool.starmap(self.correctForImage, tqdm([(fn, fn) for fn in fns]))
        # Use Topaz Sharpen AI to enhance resolution
        command = [TOPAZ_BIN] + fns
        process = Popen(command)
        process.wait()

    # Extract frames from the video using ffmpeg, and then perform correction for each frame (in place)
    # Note the video here could be exported from Premiere or other software, and not necessarily the
    # out-of-body mp4 files. So even RAW could be supported (indirectly).
    # We also give another example of directly reading in the video file (not RAW though, could be ALL-I)
    # before color grading, and invoke ffmpeg to do color grading.
    def correctForVideo(self, videofn, outdir, pool):
        if not exists(outdir):
            mkdir(outdir)
        # Example 1: don't do color grading
        # ffmpegCommand = ['ffmpeg', '-i', videofn, '-qscale:v', '2', join(outdir, "%5d.png")]
        # Example 2: do color grading. Change the cube file path to your case.
        # Cube files can be downloaded from Canon website.
        ffmpegCommand = ['ffmpeg', '-i', videofn, '-qscale:v', '2', '-vf', 'lut3d=BT2020_CanonLog3-to-BT709_WideDR_33_FF_Ver.2.0.cube', join(outdir, "%5d.png")]
        exe = Popen(ffmpegCommand)
        exe.wait()
        # Also extract the audio to be combined to the final video
        ffmpegAudioCommand = ['ffmpeg', '-i', videofn, '-vn', '-acodec', 'copy', join(outdir + '_audio.aac')]
        exe = Popen(ffmpegAudioCommand)
        exe.wait()
        # Perform the mapping and adjustment (slow) in parallel
        fns = [join(outdir, x) for x in listdir(outdir)]
        pool.starmap(self.correctForImage, tqdm([(fn, fn) for fn in fns]))
        #pool.starmap(self.correctForImage, tqdm([(fn, fn, True) for fn in fns]))
        # Get an initial version without sharpening for quick review
        command = ['ffmpeg', '-r', '30', '-i', f'{outdir}/%5d.png', '-i', f'{outdir}_audio.aac', '-c:v', 'libx264', '-vf', 'scale=8192x4096', '-preset', 'fast', '-crf', '18', '-x264-params', 'mvrange=511', '-maxrate', '100M', '-bufsize', '25M', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-movflags', 'faststart', videofn.lower().replace('.mp4', '_VR.mp4')]
        exe = Popen(command)  # We don't need to wait here

    def correctAllVideos(self):
        fns = glob('*.mp4')
        for fn in fns:
            print(f'Processing {fn}...')
            self.correctForVideo(fn, f'{fn.replace(".mp4", "")}_Frames')

if __name__ == '__main__':
    # We don't have a command line interface for now to provide maximum 
    # efficiency (e.g. no need to intialize FisheyeToEquirectangular every time)
    # Sample usage:
    procCount = min(56, cpu_count() // 2)
    print(f'Creating a process pool with {procCount} processed...')
    pool = Pool(procCount)
    converter = FisheyeToEquirectangular()
    #converter.correctForImage('./tmpframes2/00001.png', './tmpframes2/00001_corrected.png')
    #converter.correctForVideo('../VRVideoRaw/IMG_3880.MP4', 'tmpframes3880')
    #converter.correctAllImages()
    #for i in list(reversed(range(4014, 4019))):
        #fn = f'IMG_{i}.MP4'
        #newfn = f'IMG_{i}'
        #converter.correctForVideo(fn, newfn, pool)
    converter.correctAllImages(pool)
    #converter.correctAllVideos()
