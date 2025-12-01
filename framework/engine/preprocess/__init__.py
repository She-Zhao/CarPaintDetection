from .preprocess_api import Preprocessor  # 从当前目录的 run.py 导入 run_preprocess
from .stitched2 import stitch_images

__all__ = ["Preprocessor", "stitch_images"]     # 可选：定义外部可导入的符号