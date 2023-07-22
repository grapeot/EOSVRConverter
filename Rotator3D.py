import numpy as np
import cv2
from math import atan2, sqrt, sin, cos, acos, asin
from tqdm import tqdm
from os.path import exists

# Input is an LR 180 image from Canon VR
# Output is an UD 360 image
class Rotator3D:
    def __init__(self):
        self.h = 4096
        self.w = 4096
        self.data = None
        self.npfn = 'rotate.npy'

    # Currently hard coded 
    def constructTransformMatrix(self):
        print('Constructing transform matrix...')
        h = self.h
        w = self.w
        xs = np.zeros((h, w * 2), 'float32')
        ys = np.zeros((h, w * 2), 'float32')
        cx = w // 2
        cy = h // 2
        rx = w // 2
        ry = h // 2
        # rotMat = np.asarray([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        for lat in tqdm(range(h // 2)):
            for lon in range(w * 2):
                latF = (h - lat) / h * np.pi - np.pi / 2
                lonF = lon / w * np.pi
                #latF = 45 / 180 * np.pi
                #lonF = 180 / 180 * np.pi
                x = cos(latF) * cos(np.pi - lonF)
                y = cos(latF) * sin(np.pi - lonF)
                z = sin(latF)
                #print()
                #print(x, y, z)
                x, y, z = x, z, -y
                #print(x, y, z)
                oldLatF = asin(z)
                oldLonF = np.pi - atan2(y, x)
                #print(latF, lonF, oldLatF, oldLonF)
                #import pdb; pdb.set_trace()
                oldLat = h - (oldLatF + np.pi / 2) / np.pi * h
                oldLon = oldLonF / np.pi * w
                xs[lat, lon] = oldLon
                ys[lat, lon] = oldLat
        self.data = [xs, ys]
        np.save(self.npfn, self.data)

    def loadOrConstructTransformMatrix(self, forceReconstruct=False):
        if exists(self.npfn) and not forceReconstruct:
            self.data = np.load(self.npfn)
        else:
            self.constructTransformMatrix()

    def transformOneImage(self, img):
        h, w, c = img.shape
        assert(h == self.h)
        assert(w == self.w)
        if self.data is None:
            self.loadOrConstructTransformMatrix()
        img2 = cv2.remap(
            img, self.data[0], self.data[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        img2[h // 2:, ...] = 0
        return img2

    def transform(self, img):
        #img = cv2.imread(imgfn)
        h, w, c = img.shape
        assert(h == self.h)
        assert(w == self.w * 2)
        leftImg = img[:, :w // 2, :]
        rightImg = img[:, w // 2:, :]
        leftImg = self.transformOneImage(leftImg)
        rightImg = self.transformOneImage(rightImg)
        newimg = np.vstack((leftImg, rightImg))
        return newimg


if __name__ == '__main__':
    obj = Rotator()
    img = obj.transform('./IMG_9649/00001.png')
    cv2.imwrite('tmp.png', img)
