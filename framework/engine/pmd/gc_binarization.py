"""
文件: graycode_binarization.py
功能: 格雷码图像二值化处理模块，同时将条纹级次的计算迁移到了这部分。
依赖: 
- OpenCV (图像读取/保存)
- PyTorch (GPU加速计算)
- NumPy (数组处理)
- wrapped_phase.WrappedPhase (阈值计算)

典型用法:
>>> from graycode_binarization import Binariization
>>> processor = Binariization(datapath='./graycode_images/')
>>> binary_codes = processor.get_Binary_wph(offset=15)  # 获取二值化格雷码
>>> series, series1 = processor.get_series()  # 计算级次图像
"""
import os
import time
import cv2 as cv
import numpy as np
import torch
from wrapped_phase import WrappedPhase
from typing import List, Union, Tuple  # 添加导入
from pathlib import Path

'''
功能：二值化格雷码
重点在于阈值的计算，目前采用四幅图像求各个点阈值的方法

'''

class Binariization():
    """格雷码图像二值化处理器
    
    核心算法流程:
    1. 加载5幅格雷码图像 (gc0.png ~ gc4.png)
    2. 基于相移图计算自适应阈值
    3. 执行二值化处理
    4. 生成级次编码图像

    属性说明:
        graycodes (torch.Tensor): 原始格雷码图像GPU张量，形状(5,H,W)
            - 设备: 自动选择CUDA GPU或CPU
            - 数值范围: 原始灰度值(0~255)
        _binary_done (bool): 二值化完成标志
    """

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
        """初始化格雷码处理器
        Args:
            datapath: 格雷码图像目录路径，需包含gc0.png~gc4.png
            imgs: 直接传入的5张格雷码图像数组，形状需一致
            th1-th5: 各格雷码图像的阈值权重系数，默认1.0
        """        
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
        """加载图像数据到GPU张量
        Returns:
            torch.Tensor: 堆叠后的5张格雷码图，形状(5, H, W)
        """
        if self.imgs is not None:
            return self._convert_to_tensor(self.imgs[:5])
        return self._load_from_datapath()

    def _convert_to_tensor(self, img_arrays: np.ndarray) -> torch.Tensor:
        """NumPy数组转GPU张量
        Args:
            img_arrays: 输入图像数组列表，长度必须为5
        Returns:
            torch.Tensor: 在GPU上的张量，保持uint8类型
        """
        return torch.stack([
            torch.as_tensor(arr, dtype=torch.uint8, device=self.device)
            for arr in img_arrays
        ], dim=0)

    def _load_from_datapath(self) -> torch.Tensor:
        """从文件系统加载格雷码图
        Raises:
            FileNotFoundError: 当缺少图像文件时抛出
        """
        img_arrays = []
        for i in range(self.n):
            img_path = os.path.join(self.datapath, f"gc{i}.png")
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE).astype(np.float32)
            img_arrays.append(img)
        return self._convert_to_tensor(img_arrays)

    def get_threshold(self):
        """计算自适应阈值图
        基于相移图计算每个像素的灰度平均值作为阈值基准

        Returns:
            torch.Tensor: 阈值图，形状(H,W)，设备同输入
        """
        if self.datapath:
            wph = WrappedPhase(datapath = self.datapath)
        elif self.imgs:
            wph = WrappedPhase(imgs = self.imgs)
        else:
            raise RuntimeError('至少传入一种参数！图像所在路径或图像数组')
        
        return torch.mean(wph.sins, dim=0).round().to(torch.uint8)     #tensor,cuda:0

    def get_Binary_wph(self, offset:int=20):
        """执行二值化处理
        Args:
            offset: 阈值补偿量(0-255)，值越小二值化结果越白
            
        Returns:
            torch.Tensor: 二值化后的格雷码，形状(5,H,W)
                        数值为0或255，设备同输入
        """
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
        """生成级次编码图像
        通过异或运算将格雷码转换为二进制级次

        Returns:
            Tuple: (series, series1) 两个级次图
                   - series: 标准级次图
                   - series1: 位移补偿级次图
                   数值范围: 0~31 (5位编码)
        """
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
