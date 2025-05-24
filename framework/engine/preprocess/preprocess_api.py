from typing import List, Union
import numpy as np
import torch

# 对外暴漏的接口
def run_preprocess(
    imgs: List[List[np.ndarray]]
) -> List[np.ndarray]:
    """调用图像拼接算法，待补充

    Args:
        imgs: List[np.ndarray]，相机拍摄的源图像，所以是ndarray，而不是tensor
    
    Returns:
        图像拼接的结果，估计是List[np.ndarray]，里面每个np.ndarray是拼接后的四张图像。
    """
    return imgs
        
if __name__ == "__main__":
    run_preprocess()
