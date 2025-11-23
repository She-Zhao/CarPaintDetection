from typing import List, Union
import numpy as np
import torch
from . import stitch_images
import json
from importlib.resources import files, as_file

# 对外暴漏的接口
def run_preprocess(raw_imgs: List[List[np.ndarray]]) -> List[np.ndarray]:
    """调用图像拼接算法，待补充

    Args:
        imgs:List[List[np.ndarray]]，每个List[np.ndarray]是相机拍摄的一组原始图像（gc0~sin3）。
    
    Returns:
        图像拼接的结果，估计是List[np.ndarray]，里面每个np.ndarray是n个相机拼接之后的结果(一张大图)。
    """
    # 安全访问包内资源
    ref = files("framework.data").joinpath("H.json")
    with as_file(ref) as f:
        H = np.array(json.load(f))
    stitched_img = stitch_images(raw_imgs[0], raw_imgs[1], H)
    processed_imgs = stitched_img
    return processed_imgs
        
if __name__ == "__main__":
    run_preprocess()
