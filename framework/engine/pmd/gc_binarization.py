import glob
import os
import time
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from wrapped_phase import WrappedPhase
from typing import List, Union, Tuple  # 添加导入
from pathlib import Path

'''
功能：二值化格雷码
重点在于阈值的计算，目前采用四幅图像求各个点阈值的方法

'''

class Binariization():
    '''对格雷码图像进行二值化

    get_GC_images:读取格雷码图像,J_array.shape(4,2048,2448)
    get_Binary_adaptive:自适应阈值法二值化

    Attributes:
        datapath:存储格雷码图像的路径
        imgs: 是否直接输入图像
        n:n副格雷码图像

    '''

    def __init__(
        self,
        datapath: str = None,
        imgs: List[np.ndarray] = None,  # 或者 Tuple[np.ndarray, ...]
        th1: float = 1.0,
        th2: float = 1.0,
        th3: float = 1.0,
        th4: float = 1.0,
        th5: float = 1.0,
    ):
        self.datapath = datapath
        self.imgs = imgs if imgs is not None else imgs
        self.th1 = th1
        self.th2 = th2
        self.th3 = th3
        self.th4 = th4
        self.th5 = th5
        self.n = 5
        self._binary_done = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.graycodes = self._load_image_data()        # 原图像
        

    def _load_image_data(self) -> torch.Tensor:
        """加载图像数据到GPU张量"""
        if self.imgs is not None:
            return self._convert_to_tensor(self.imgs[:5])
        return self._load_from_datapath()

    def _convert_to_tensor(self, img_arrays: np.ndarray) -> torch.Tensor:
        """将NumPy数组批量转换为GPU张量"""
        return torch.stack([
            torch.as_tensor(arr, dtype=torch.uint8, device=self.device)
            for arr in img_arrays
        ], dim=0)

    def _load_from_datapath(self) -> torch.Tensor:
        """从文件系统加载相移图"""
        img_arrays = []
        for i in range(self.n):
            img_path = os.path.join(self.datapath, f"gc{i}.png")
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE).astype(np.float32)
            img_arrays.append(img)
        return self._convert_to_tensor(img_arrays)

    def get_threshold(self):
        '''利用四幅相移图计算阈值'''
        if self.datapath:
            wph = WrappedPhase(datapath = self.datapath)
        elif self.imgs:
            wph = WrappedPhase(imgs = self.imgs)
        else:
            raise RuntimeError('至少传入一种参数！图像所在路径或图像数组')
        
        return torch.mean(wph.sins, dim=0).round().to(torch.uint8)     #tensor,cuda:0

    def get_Binary_wph(self, offset:int=20):
        '''利用四幅相移图求阈值，将格雷码图像二值化处理
        Args:增加了一个补偿offset，该值越小，阈值越小，图像越白
        return:四幅二值化后格雷码图像，tensor,cuda:0
        '''
        threshold = self.get_threshold()        #threshold.device：cuda:0

        self.graycodes[0][self.graycodes[0] <= (threshold + self.th1*offset)] = 0
        self.graycodes[0][self.graycodes[0] > (threshold + self.th1*offset)] = 255

        self.graycodes[1][self.graycodes[1] <= (threshold + self.th2*offset)] = 0
        self.graycodes[1][self.graycodes[1] > (threshold + self.th2*offset)] = 255

        self.graycodes[2][self.graycodes[2] <= (threshold + self.th3*offset)] = 0
        self.graycodes[2][self.graycodes[2] > (threshold + self.th3*offset)] = 255

        self.graycodes[3][self.graycodes[3] <= (threshold + self.th4*offset)] = 0
        self.graycodes[3][self.graycodes[3] > (threshold + self.th4*offset)] = 255

        self.graycodes[4][self.graycodes[4] <= (threshold + self.th5*offset)] = 0
        self.graycodes[4][self.graycodes[4] > (threshold + self.th5*offset)] = 255
        
        self._binary_done = True 

        # if __name__ != "__main__":
        #     self.graycodes1 = self.graycodes.cpu().numpy()
        #     for u in range(self.n):
        #         #self.graycodes = self.graycodes[u].to(torch.uint8)
        #         cv.imwrite('.\output' + '\Binarized_GC-' + str(u) + ".png", self.graycodes1[u])

        return self.graycodes

    def get_series(self):
        '''将格雷码转换为级次图像，并保存级次图像

        Args:
            self.graycodes: 相机格雷码黑白(0 255)图像, tensor,cuda:0, .shape(4,row,col)

        Returns:
            series，series: 周期级次图像,tensor,cuda:0
        '''
        if not self._binary_done:
            self.get_Binary_wph()

        self.graycodes = (self.graycodes // 255)

        rows, cols = self.graycodes[0].shape
        series = torch.zeros((rows, cols), dtype=torch.uint8, device=self.device)
        series1 = torch.zeros((rows, cols), dtype=torch.uint8, device=self.device)
        bin_img = torch.zeros_like(self.graycodes, device=self.device)
        bin_img[0] = self.graycodes[0]

        for i in range(1, self.n):
            bin_img[i] = torch.bitwise_xor(self.graycodes[i], bin_img[i - 1])

        for j in range(0, self.n):
            if self.n - j - 2 >= 0:
                series += 2**(self.n - j - 2) * bin_img[j]
            if self.n - j - 1 >= 0:
                series1 += 2**(self.n - j - 1) * bin_img[j]

        series1 = torch.floor((series1 + 1) / 2)

        # series_scale = (series * (255 / 2**(self.n - 1) - 1)).to(torch.uint8)
        # cv.imwrite('.\output\series.png', series_scale.cpu().numpy())

        return series, series1    
    

if __name__ == "__main__":
    root_dir = r"/home/nvidia/Project/pmd_adjust/test_imgs"
    for idx, dir in enumerate(os.listdir(root_dir)):
        datapath = os.path.join(root_dir, dir)
        t0 = time.time()
        img_list = [cv.imread(os.path.join(datapath, img), cv.IMREAD_GRAYSCALE) 
                    for img in sorted(os.listdir(datapath))]
        
        bgc = Binariization(imgs=img_list)
        gc = bgc.get_Binary_wph(12)
        t1 = time.time()
        print(f">>>>>{t1-t0}")
        gc=gc.cpu().numpy()

        for u in range(bgc.n):
            #gc[u].astype(np.uint8)
            save_dir = os.path.join(Path(__file__).parent.name, f'output', f"{idx}")
            save_path = os.path.join(save_dir, f'Binarized_GC-{u}.png')
            os.makedirs(save_dir, exist_ok=True)
            cv.imwrite(save_path, gc[u])


    # datapath = r"/home/nvidia/Project/pmd_adjust/test_imgs/pos1"
    # t0 = time.time()
    # bgc = Binariization(datapath)
    # gc = bgc.get_Binary_wph(12)
    # t1 = time.time()
    # print(f">>>>>{t1-t0}")
    # gc=gc.cpu().numpy()

    # for u in range(bgc.n):
    #     #gc[u].astype(np.uint8)
    #     save_path = os.path.join(Path(__file__).parent.name, 'output', f'Binarized_GC-{u}.png')
    #     cv.imwrite(save_path, gc[u])
