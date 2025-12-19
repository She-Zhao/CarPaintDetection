import os
import cv2 as cv
from framework.engine.pmd.pmd_api import PMDprocessor

if __name__ == "__main__":
    pmdprocessor = PMDprocessor()
    datapath1 = r'D:\Project\_New_System\test\pos1'
    datapath2 = r'D:\Project\_New_System\test\pos2'
    imgs1 = [cv.imread(os.path.join(datapath1, img), cv.IMREAD_GRAYSCALE) 
                for img in sorted(os.listdir(datapath1))]
    imgs2 = [cv.imread(os.path.join(datapath2, img), cv.IMREAD_GRAYSCALE) 
                for img in sorted(os.listdir(datapath2))] 
    imgs = [imgs1, imgs2]  

    absphases = pmdprocessor(imgs)
    for idx, absphase in enumerate(absphases):
        # 确保数据在CPU上并转换为numpy
        phase_np = absphase.numpy()
        # 缩放到0-255并转换为uint8
        phase_np = cv.normalize(phase_np, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        cv.imwrite(os.path.join(datapath1, f'abs{idx}.png'), phase_np)