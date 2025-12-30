from typing import List, Union
import numpy as np
from .stitched2 import stitch_images
import cv2

class Preprocessor:
    """图像预处理器"""

    def __call__(self, raw_imgs: List[List[np.ndarray]], H_matrix: np.ndarray) -> List[np.ndarray]:
        """调用图像拼接算法，待补充

        Args:
            imgs:List[List[np.ndarray]]，每个List[np.ndarray]是相机拍摄的一组原始图像（gc0~sin3）。
        
        Returns:
            图像拼接的结果，List[np.ndarray]，里面每个np.ndarray是n个相机拼接之后的结果(一张大图)。
        """
        # stitched_img = stitch_images(raw_imgs[0], raw_imgs[1], H_matrix)     # raw_imgs第一个idx对应相机,第二个idx对应第几个图像
        # processed_imgs = stitched_img
        resize_imgs = [cv2.resize(img, (2448, 2048)) for img in raw_imgs[1]]
        return resize_imgs
