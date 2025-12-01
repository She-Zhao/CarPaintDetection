from typing import List, Union
import numpy as np
from .stitched2 import stitch_images
import json

class Preprocessor:
    """图像预处理器"""
    def __init__(self):
        self.H_matrix = self._load_H_matrix()

    def _load_H_matrix(self):
        """加载单应性矩阵"""
        with open('data/param.json') as f:
            param = json.load(f)
            H_matrix = param['H_matrix']
            print(f"🚀单应性矩阵加载成功! H = \n{H_matrix}")
            
        return H_matrix

    def __call__(self, raw_imgs: List[List[np.ndarray]]) -> List[np.ndarray]:
        """调用图像拼接算法，待补充

        Args:
            imgs:List[List[np.ndarray]]，每个List[np.ndarray]是相机拍摄的一组原始图像（gc0~sin3）。
        
        Returns:
            图像拼接的结果，估计是List[np.ndarray]，里面每个np.ndarray是n个相机拼接之后的结果(一张大图)。
        """
        stitched_img = stitch_images(raw_imgs[0], raw_imgs[1], self.H_matrix)
        processed_imgs = stitched_img
        return processed_imgs


if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor()
