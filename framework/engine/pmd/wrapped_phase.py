"""
文件: pmd.py
功能: 相位解包裹核心算法模块，基于四步相移法计算包裹相位
依赖: 
- PyTorch (GPU加速)
- OpenCV (图像读取/保存)
- NumPy (数组处理)

典型用法:
>>> from engine.pmd import WrappedPhase
>>> processor = WrappedPhase(datapath='./phase_images/')
>>> wrapped_phase = processor.computeWrappedphase()  # 返回[0, 2π]的相位张量
>>> # 或直接传入图像数组，速度更快！
>>> imgs = [cv.imread(f'sin{i}.png', cv.IMREAD_GRAYSCALE) for i in range(4)]
>>> processor = WrappedPhase(imgs=imgs)
"""
import math
import time
import os
import numpy as np
import cv2 as cv
import torch
from pathlib import Path
from typing import Optional, Union, List  # 添加导入

'''
功能：计算折叠相位
按下s键自动保存为png格式。不要在python的页面保存，这样保存的图像不清晰
最后编辑时间：2023.8.25.21
'''

class WrappedPhase():
    """四步相移法相位解包裹处理器
    
    核心算法流程:
    1. 加载4张相移图 (sin0.png ~ sin3.png)
    2. 计算各像素点相位值 (0~2π)
    3. 应用相位跳变滤波 (PJE Filter)
    
    属性说明:
        sins (torch.Tensor): 四张相移图的GPU张量，形状为(4, H, W)
            - 存储顺序: [sin0, sin1, sin2, sin3]
            - 设备: 自动选择CUDA GPU或CPU
            - 数值范围: 原始灰度值(0~255)
    """
    def __init__(
        self,
        datapath: Optional[str] = None,
        imgs: List[np.ndarray] = None,
        width: int = 2448,
        height: int = 2048,
        jump_step: int = 4,
    ) -> None:
        """初始化相位处理器
        Args:
            datapath: 相移图目录路径，需包含sin0.png~sin3.png
            imgs: 直接传入的4张相移图数组，形状需一致
            width: 图像宽度(像素)，默认2448
            height: 图像高度(像素)，默认2048
            jump_step: 相位跳变检测步长(1~10)，默认4
        """
        self.datapath = datapath
        self.imgs = imgs if imgs is not None else None
        self.width = width
        self.height = height
        self.jump_step = jump_step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sins = self._load_image_data()

    def _load_image_data(self) -> torch.Tensor:
        """加载图像数据到GPU张量
        Returns:
            torch.Tensor: 堆叠后的4张相移图，形状(4, H, W)
        """
        if self.imgs is not None:
            return self._convert_to_tensor(self.imgs[5:9])
        return self._load_from_datapath()

    def _convert_to_tensor(self, img_arrays: np.ndarray) -> torch.Tensor:
        """NumPy数组转GPU张量
        Args:
            img_arrays: 输入图像数组列表，长度必须为4
        Returns:
            torch.Tensor: 在GPU上的张量
        """
        return torch.stack([
            torch.as_tensor(arr, dtype=torch.float32, device=self.device)
            for arr in img_arrays
        ], dim=0)

    def _load_from_datapath(self) -> torch.Tensor:
        """从文件系统加载相移图
        Raises:
            FileNotFoundError: 当缺少相移图时抛出
        """
        img_arrays = []
        for i in range(4):
            img_path = os.path.join(self.datapath, f"sin{i}.png")
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE).astype(np.float32)
            img_arrays.append(img)
        return self._convert_to_tensor(img_arrays)
    
    def pje_filter(self, pha):
        """相位跳变边缘滤波
        Args:
            pha: 输入相位图，范围[0, 2π]
        Returns:
            torch.Tensor: 滤波后的相位图
        """        
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
        """计算包裹相位
        算法步骤:
        1. 根据四步相移公式分象限计算相位
        2. 应用PJE滤波消除跳变边缘
        
        Returns:
            torch.Tensor: 解包裹相位图，范围[0, 2π]，设备同输入
        """
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
