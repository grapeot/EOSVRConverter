from glob import glob
from os.path import join
from subprocess import Popen
from sys import exit

TOPAZ_BIN = r"C:\Program Files\Topaz Labs LLC\Topaz Sharpen AI\Topaz Sharpen AI.exe"
TOPAZ_INSTANCE = 8
WORK_DIRS = [
    f'./IMG_{i}'
    for i in [4055, 4056, 4058, 4059, 4060, 4061, 4065, 4066, 4068, 4069, 4070, 4075]
]
OUT_FNS = [
    f'{d}_VR_Enlarge.mp4'
    for d in WORK_DIRS
]

if __name__ == '__main__':
    # Invoke Topaz Sharpen AI to enhance clarity
    # Use TOPAZ_INSTANCE instances by default
    fns = []
    for d in WORK_DIRS:
        fns += glob(join(d, '*.png'))
    n = len(fns)
    batchSize = n // TOPAZ_INSTANCE
    fnsForEachInstance = [fns[batchSize * i: batchSize * (i + 1)] for i in range(TOPAZ_INSTANCE)]
    if n > batchSize * TOPAZ_INSTANCE - 1:
        fnsForEachInstance[-1].extend(fns[batchSize * TOPAZ_INSTANCE:])
    print([len(f) for f in fnsForEachInstance])
    isContinue = input('Continue (y/n)?')
    if isContinue != 'y':
        exit(1)
    processes = []
    for f in fnsForEachInstance:
        command = [TOPAZ_BIN] + f
        processes.append(Popen(command))
    for p in processes:
        p.wait()
    
    # Invoke ffmpeg to recombine the video
    # In this case, SharpenAI was launched in plugin mode, and will overwrite the original files
    # So we don't need to do any renaming
    processes = []
    for workdir, outfn in zip(WORK_DIRS, OUT_FNS):
        command = ['ffmpeg', '-r', '30', '-i', f'{workdir}/%5d.png', '-i', f'{workdir}_audio.aac', '-c:v', 'libx264', '-vf', 'scale=8192x4096', '-preset', 'fast', '-crf', '18', '-x264-params', 'mvrange=511', '-maxrate', '100M', '-bufsize', '25M', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-movflags', 'faststart', outfn]
        p = Popen(command)
        processes.append(p)
        p.wait()
