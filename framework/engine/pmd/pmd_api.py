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
from functools import partial
from typing import List, Union, Optional

def run_pmd_single(imgs: List[np.ndarray]) -> Optional[torch.Tensor]:
    """单组图像计算绝对相位
    
    Args:
        imgs: List[np.ndarray]，相机拍摄的源图像，所以是ndarray，而不是tensor
        后面会改成从preprocess处得到的结果，也需要是np.ndarry
        
    Returns:
        解包裹后的绝对相位图(GPU Tensor)
    """    
    try:
        if torch.cuda.is_available():
            torch.cuda.init()  # 更规范的初始化
            torch.zeros(1).cuda()
            torch.cuda.empty_cache()  # 清理残留缓存

        W = WrappedPhase(imgs=imgs)
        wph = W.computeWrappedphase()

        B = Binarization(imgs=imgs, th1=th1, th2=th2, th3=th3, th4=th4, th5=th5) 
        series, series1 = B.get_series()

        U = Unwrappedphase()
        absphase = U.get_absphase(wph, series, series1)
        return absphase
    
    except Exception as e:
        print(f"处理失败: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
   
def run_pmd_parallel(imgs_list: List[List[np.ndarray]]) -> List[Optional[torch.Tensor]]:
    """多组图像并行处理
    
    Args:
        imgs_list: 多组图像输入: List[List[np.ndarray]]
        
    Returns:
        绝对列表(每个元素为CPU Tensor)
    """    
    if torch.cuda.is_available():
        with multiprocessing.Pool(processes=2) as pool:         # GPU任务使用进程池（避免GIL限制）
            return pool.map(run_pmd_single, imgs_list)
        
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:     # CPU任务使用线程池
            return list(executor.map(run_pmd_single, imgs_list))

# 对外暴漏的接口
def run_pmd(
    imgs: Union[List[np.ndarray], List[List[np.ndarray]]]
) -> List[torch.Tensor]:
    """PMD算法主入口，处理单组/多组图像输入。

    Args:
        imgs: 单组图像列表或多组图像列表的列表。
    
    Returns:
        绝对相位列表（CPU Tensor，已归一化范围0-255）
    """
    if not isinstance(imgs, List):
        raise TypeError(f"输入必须是 list，但得到 {type(imgs)}")
    
    if isinstance(imgs[0], np.ndarray):
        return [run_pmd_single(imgs)] 
    elif isinstance(imgs[0], List):
        return run_pmd_parallel(imgs)
        
if __name__ == "__main__":
    datapath1 = r'D:\Project\_New_System\test\pos1'
    datapath2 = r'D:\Project\_New_System\test\pos2'
    imgs1 = [cv.imread(os.path.join(datapath1, img), cv.IMREAD_GRAYSCALE) 
                for img in sorted(os.listdir(datapath1))]
    imgs2 = [cv.imread(os.path.join(datapath2, img), cv.IMREAD_GRAYSCALE) 
                for img in sorted(os.listdir(datapath2))] 
    imgs = [imgs1, imgs2]  

    absphases = run_pmd(imgs)
    for idx, absphase in enumerate(absphases):
        # 确保数据在CPU上并转换为numpy
        phase_np = absphase.numpy()
        # 缩放到0-255并转换为uint8
        phase_np = cv.normalize(phase_np, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        cv.imwrite(os.path.join(datapath1, f'abs{idx}.png'), phase_np)



    # datapath = r'D:\Project\_New_System\test\pos2'
    
    # imgs = [cv.imread(os.path.join(datapath, img), cv.IMREAD_GRAYSCALE) 
    #             for img in sorted(os.listdir(datapath))] 
            
    # absphase = main(imgs)
    # cv.imwrite(os.path.join(datapath, 'abs.png') , absphase.cpu().numpy())
