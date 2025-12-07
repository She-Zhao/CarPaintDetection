from pathlib import Path
import os
import time
import numpy as np
import cv2 as cv
import torch
from framework.engine.pmd.wrapped_phase import WrappedPhase
from framework.engine.pmd.gc_binarization import Binarization
from framework.engine.pmd.unwrapped_phase import Unwrappedphase
from typing import List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from typing import List, Union, Optional

class PMDprocessor:
    def __init__(self, max_workers=2):
        self.max_workers = max_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PMD] 初始化完成，使用设备 {self.device}, 并发数: {self.max_workers}")
        
    def __call__(self, imgs, th_list):
        if not isinstance(imgs, List):
            raise TypeError(f"输入必须是 list，但得到 {type(imgs)}")
        
        if isinstance(imgs[0], np.ndarray):
            return [self._process_single(imgs, th_list)] 
        elif isinstance(imgs[0], List):
            return self._process_multi(imgs, th_list)
    
    def _process_single(self, imgs: List[np.ndarray], th_list: List) -> Optional[torch.Tensor]:
        try:
            W = WrappedPhase(imgs=imgs)
            wph = W.computeWrappedphase()

            B = Binarization(imgs=imgs, thresholds=th_list) 
            series, series1 = B.get_series()

            U = Unwrappedphase()
            absphase = U.get_absphase(wph, series, series1)
            return absphase 
        except Exception as e:
            print(f"处理失败: {str(e)}")
               
    def _process_multi(self, imgs_list: List[List[np.ndarray]]) -> List[Optional[torch.Tensor]]:
        if torch.cuda.is_available():
            with multiprocessing.Pool(processes=2) as pool:         # GPU任务使用进程池（避免GIL限制）
                return pool.map(self._process_single, imgs_list)
            
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:     # CPU任务使用线程池
                return list(executor.map(self._process_single, imgs_list))    

# def run_pmd_single(imgs: List[np.ndarray]) -> Optional[torch.Tensor]:
#     """单组图像计算绝对相位
    
#     Args:
#         imgs: List[np.ndarray]，相机拍摄的源图像，所以是ndarray，而不是tensor
#         后面会改成从preprocess处得到的结果，也需要是np.ndarry
        
#     Returns:
#         解包裹后的绝对相位图(GPU Tensor)
#     """    
#     try:
#         if torch.cuda.is_available():
#             torch.cuda.init()  # 更规范的初始化
#             torch.zeros(1).cuda()
#             torch.cuda.empty_cache()  # 清理残留缓存

#         W = WrappedPhase(imgs=imgs)
#         wph = W.computeWrappedphase()

#         B = Binarization(imgs=imgs, th1=th1, th2=th2, th3=th3, th4=th4, th5=th5) 
#         series, series1 = B.get_series()

#         U = Unwrappedphase()
#         absphase = U.get_absphase(wph, series, series1)
#         return absphase
    
#     except Exception as e:
#         print(f"处理失败: {str(e)}")
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
   
# def run_pmd_parallel(imgs_list: List[List[np.ndarray]]) -> List[Optional[torch.Tensor]]:
#     """多组图像并行处理
    
#     Args:
#         imgs_list: 多组图像输入: List[List[np.ndarray]]
        
#     Returns:
#         绝对列表(每个元素为CPU Tensor)
#     """    
#     if torch.cuda.is_available():
#         with multiprocessing.Pool(processes=2) as pool:         # GPU任务使用进程池（避免GIL限制）
#             return pool.map(run_pmd_single, imgs_list)
        
#     else:
#         with ThreadPoolExecutor(max_workers=2) as executor:     # CPU任务使用线程池
#             return list(executor.map(run_pmd_single, imgs_list))

# # 对外暴漏的接口
# def run_pmd(
#     imgs: Union[List[np.ndarray], List[List[np.ndarray]]]
# ) -> List[torch.Tensor]:
#     """PMD算法主入口，处理单组/多组图像输入。

#     Args:
#         imgs: 单组图像列表或多组图像列表的列表。
    
#     Returns:
#         绝对相位列表（CPU Tensor，已归一化范围0-255）
#     """
#     if not isinstance(imgs, List):
#         raise TypeError(f"输入必须是 list，但得到 {type(imgs)}")
    
#     if isinstance(imgs[0], np.ndarray):
#         return [run_pmd_single(imgs)] 
#     elif isinstance(imgs[0], List):
#         return run_pmd_parallel(imgs)
