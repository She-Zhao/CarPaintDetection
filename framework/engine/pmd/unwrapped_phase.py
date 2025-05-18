"""
文件: phase_unwrapping.py
功能: 相位解包裹核心模块，将包裹相位转换为绝对相位
依赖:
- PyTorch (GPU加速)
- OpenCV (图像读取)
- numpy (数组处理)
- gc_binarization.Binariization (格雷码处理)
- wrapped_phase.WrappedPhase (包裹相位计算)

典型用法:
>>> from phase_unwrapping import Unwrappedphase
>>> unwrapper = Unwrappedphase()
>>> abs_phase = unwrapper.get_absphase(wrapped_phase, series, series1)  # 输入为GPU张量
"""
#import inspect
from pathlib import Path
import os
import time
import numpy as np
import cv2 as cv
import torch
from gc_binarization import Binariization
from wrapped_phase import WrappedPhase 

class Unwrappedphase():
    """相位解包裹处理器
    
    核心算法流程:
    1. 接收包裹相位图(0-2π)和格雷码级次图
    2. 根据相位值区间选择不同的级次补偿策略
    3. 生成连续绝对相位场

    属性说明:
        device (torch.device): 自动选择CUDA GPU或CPU
        n (int): 格雷码图像数量，固定为5
    """
    def __init__(self):
        self.n = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_absphase(self, wrappedphase, series, series1):
        """计算绝对相位
        
        Args:
            wrappedphase: 包裹相位图，范围[0, 2π]，GPU张量
            series: 标准级次图，GPU张量
            series1: 补偿级次图，GPU张量.与series均在二值化文件中进行计算。

        Returns:
            torch.Tensor: 绝对相位场(无范围限制)，设备同输入
        """

        unwrapped_pha = torch.zeros_like(series, dtype=torch.float32, device=self.device)

        idx1 = wrappedphase <= (torch.pi / 2)
        idx2 = (wrappedphase > (torch.pi / 2)) & (wrappedphase < (3 * torch.pi / 2))
        idx3 = wrappedphase >= (3 * torch.pi / 2)
        
        unwrapped_pha[idx1] = wrappedphase[idx1] + series1[idx1] * 2 * torch.pi
        unwrapped_pha[idx2] = wrappedphase[idx2] + series[idx2] * 2 * torch.pi
        unwrapped_pha[idx3] = wrappedphase[idx3] + (series1[idx3] - 1) * 2 * torch.pi

        absphase = ((unwrapped_pha*255)/(2**5*torch.pi)).to(torch.uint8) 
        
        return absphase

if __name__ == "__main__":
    root_dir = r"/home/nvidia/Project/pmd_adjust/test_imgs"
    if torch.cuda.is_available():
        torch.zeros(1).cuda()  # 触发CUDA初始化    
        
    for idx, dir in enumerate(os.listdir(root_dir)):
        datapath = os.path.join(root_dir, dir)
        img_list = [cv.imread(os.path.join(datapath, img), cv.IMREAD_GRAYSCALE) 
                    for img in sorted(os.listdir(datapath))]

        t0 = time.time()
        W = WrappedPhase(imgs=img_list)
        wph = W.computeWrappedphase()
        t1 = time.time()
        print(f"折叠相位时间{t1-t0}s")

        B = Binariization(imgs=img_list)
        series, series1 = B.get_series()
        t2 = time.time()
        print(f"条纹级数计算时间{t2-t1}s")

        U = Unwrappedphase() 
        absphase = U.get_absphase(wph, series, series1)
        t3 = time.time()
        print(f"绝对相位计算时间{t3-t2}s")  

        time.sleep(2)
