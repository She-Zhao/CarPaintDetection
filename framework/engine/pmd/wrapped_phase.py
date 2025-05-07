import torchvision 
import math
import time
import os
import numpy as np
import cv2 as cv
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union, List  # 添加导入

'''
功能：计算折叠相位
按下s键自动保存为png格式。不要在python的页面保存，这样保存的图像不清晰
最后编辑时间：2023.8.25.21
'''

class WrappedPhase():
    '''
    用于求解包裹相位
    '''
    def __init__(
        self,
        datapath: Optional[str] = None,
        imgs: List[np.ndarray] = None,
        width: int = 2448,
        height: int = 2048,
        jump_step: int = 4,
    ) -> None:
        """Initialize the phase unwrapping processor.

        Args:
            datapath: 相移图所在文件夹.
            imgs: 是否直接输入图像数组.
            width: 图像高度
            height: 图像宽度.
            jump_step: 判断跳变得步数(1-10).
        """
        self.datapath = datapath
        self.imgs = imgs if imgs is not None else None
        self.width = width
        self.height = height
        self.jump_step = jump_step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sins = self._load_image_data()

    def _load_image_data(self) -> torch.Tensor:
        """加载图像数据到GPU张量"""
        if self.imgs is not None:
            return self._convert_to_tensor(self.imgs[-4:])
        return self._load_from_datapath()

    def _convert_to_tensor(self, img_arrays: np.ndarray) -> torch.Tensor:
        """将NumPy数组批量转换为GPU张量"""
        return torch.stack([
            torch.as_tensor(arr, dtype=torch.float32, device=self.device)
            for arr in img_arrays
        ], dim=0)

    def _load_from_datapath(self) -> torch.Tensor:
        """从文件系统加载相移图"""
        img_arrays = []
        for i in range(4):
            img_path = os.path.join(self.datapath, f"sin{i}.png")
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE).astype(np.float32)
            img_arrays.append(img)
        return self._convert_to_tensor(img_arrays)
    
    def pje_filter(self, pha):
        N = self.height  # 相位图的高
        n = self.jump_step  # 判定是否跳变的步数，取1-10之间

        pha_temp = pha[0:(N-n), :]
        pha_temp_1 = torch.cat([pha[-1, :].unsqueeze(0), pha[0:(N-n-1), :]])
        pha_temp_n = pha[n:, :]

        idx = ((pha_temp_1 - torch.pi) * (pha_temp - torch.pi) < 0) & ((pha_temp_n - torch.pi) * (pha_temp - torch.pi) < 0)
        pha_temp[idx] = 2 * torch.pi - pha_temp[idx]
        pha[0:(N-n), :] = pha_temp

        return pha

    def computeWrappedphase(self):
        '''
        计算包裹相位,并绘制相位图,按s键保存

        Args:
        I: 相机捕捉的相位图数组
        width, height: 照片的尺寸

        return: 真实相位[0,2*pi], pha.shape(width, height), tensor, cuda:0
        '''
        i0, i1, i2, i3 = self.sins[0], self.sins[1], self.sins[2], self.sins[3]
    
        pha = torch.zeros((self.height, self.width), dtype=torch.float32, device=self.device)
        
        idx1 = (i0 == i2) & (i3 < i1)  # 四个特殊位置
        idx2 = (i0 == i2) & (i3 > i1)  # 四个特殊位置
        idx3 = (i3 == i1) & (i0 < i2)  # 四个特殊位置
        idx4 = (i3 == i1) & (i0 > i2)  # 四个特殊位置
        idx5 = (i0 > i2) & (i1 < i3)  # 第一象限
        idx6 = (i0 < i2) & (i1 < i3)  # 第二象限
        idx7 = (i0 < i2) & (i1 > i3)  # 第三象限
        idx8 = (i0 > i2) & (i1 > i3)  # 第四象限
        pha[idx1] = 3 * torch.pi / 2
        pha[idx2] = torch.pi / 2
        pha[idx3] = torch.pi
        pha[idx4] = 0
        pha[idx5] = torch.atan((i3[idx5] - i1[idx5]) / (i0[idx5] - i2[idx5]))
        pha[idx6] = torch.pi - torch.atan((i3[idx6] - i1[idx6]) / (i2[idx6] - i0[idx6]))
        pha[idx7] = torch.pi + torch.atan((i3[idx7] - i1[idx7]) / (i0[idx7] - i2[idx7]))
        pha[idx8] = 2 * torch.pi - torch.atan((i1[idx8] - i3[idx8]) / (i0[idx8] - i2[idx8]))
        
        pha = self.pje_filter(pha)  # 进行相位跳变滤波,pha-.tensor,cuda:0
        
        return pha


if __name__ == "__main__":
    root_dir = r"/home/nvidia/Project/pmd_adjust/test_imgs"
    for idx, dir in enumerate(os.listdir(root_dir)):
        datapath = os.path.join(root_dir, dir)
        img_list = [cv.imread(os.path.join(datapath, img), cv.IMREAD_GRAYSCALE) 
                    for img in sorted(os.listdir(datapath))]
        
        t0 = time.time()

        w = WrappedPhase(imgs=img_list)
        pha = w.computeWrappedphase()

        t1 = time.time()
        print(f"total time: {t1-t0}s")
        
        with torch.no_grad():
            pha_scaled = pha * (255 / (2 * torch.pi))  # 使用张量运算保持设备一致性
            pha_scaled1 = pha_scaled.cpu().numpy().astype(np.uint8)    

        pha_scaled = pha * 255 / (2 * math.pi)      # 将pha转换到为图像灰度尺度
        pha_scaled1 = pha_scaled.cpu().numpy().astype(np.uint8)
        save_path = os.path.join(Path(__file__).parent.name, 'output', f'Wrapped_Phase{idx}.png')
        cv.imwrite(save_path, pha_scaled1)



    # root_dir = r"/home/nvidia/Project/pmd_adjust/test_imgs"
    # # datapath = r"/home/nvidia/Project/pmd_adjust/test_imgs/pos1"
    # for idx, dir in enumerate(os.listdir(root_dir)):
    #     datapath = os.path.join(root_dir, dir)

    #     t0 = time.time()
    #     w = WrappedPhase(datapath)
    #     pha = w.computeWrappedphase()              # pha是真实的折叠相位
    #     with torch.no_grad():
    #         pha_scaled = pha * (255 / (2 * w.pi))  # 使用张量运算保持设备一致性
    #         pha_scaled1 = pha_scaled.cpu().numpy().astype(np.uint8)    

    #     t1 = time.time()
    #     print(t1-t0)

    #     pha_scaled = pha * 255 / (2 * math.pi)      # 将pha转换到为图像灰度尺度
    #     pha_scaled1 = pha_scaled.cpu().numpy().astype(np.uint8)
    #     save_path = os.path.join(Path(__file__).parent.name, 'output', f'Wrapped_Phase{idx}.png')
    #     cv.imwrite(save_path, pha_scaled1)
