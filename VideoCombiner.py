from glob import glob
from os.path import join
from subprocess import Popen

TOPAZ_BIN = r"C:\Program Files\Topaz Labs LLC\Topaz Sharpen AI\Topaz Sharpen AI.exe"
TOPAZ_INSTANCE = 8
WORK_DIR = './frames3892_sharpen'
OUT_FN = 'IMG_3892.mp4'

if __name__ == '__main__':
    # Invoke Topaz Sharpen AI to enhance clarity
    # Use TOPAZ_INSTANCE instances by default
    fns = glob(join(WORK_DIR, '*.png'))
    n = len(fns)
    batchSize = n // TOPAZ_INSTANCE
    fnsForEachInstance = [fns[batchSize * i: batchSize * (i + 1)] for i in range(TOPAZ_INSTANCE)]
    if n > batchSize * TOPAZ_INSTANCE - 1:
        fnsForEachInstance[-1].append(fns[batchSize * TOPAZ_INSTANCE:])
    processes = []
    for f in fnsForEachInstance:
        command = [TOPAZ_BIN] + f
        processes.append(Popen(command))
    for p in processes:
        p.wait()
    
    # Invoke ffmpeg to recombine the video
    # In this case, SharpenAI was launched in plugin mode, and will overwrite the original files
    # So we don't need to do any renaming
    command = ['ffmpeg', '-r', '30', '-i', f'{WORK_DIR}/%5d.png', '-i', f'{WORK_DIR}_audio.aac', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-x264-params', 'mvrange=511', '-maxrate', '100M', '-bufsize', '25M', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-movflags', 'faststart', OUT_FN]
    Popen(command).wait()
