from typing import List, Union
import numpy as np
import torch


def init_model():
    # 在PipelineExecutor.__init__.py中调用，初始化模型加载
    return 

# 对外暴漏的接口
def run_detect(model,
    abs_phases: List[torch.Tensor]
) -> List[List[float]]:
    """调用初始化好的模型，输出[x,y,w,h]这样一个bbox

    Args:
        imgs: 两张绝对相位图,存储在一个List中
    
    Returns:
        网络的检测结果，bbox的中心点坐标及宽和高
    """
    # load_model()
    cls = 'crater'
    x = 2.0
    y = 2.0
    w = 1.0
    h = 1.0
    conf = 0.65
    abs_phases_detected = abs_phases
    return [[cls, x, y, w, h, conf], [cls, x, y, w, h, conf]]
        
if __name__ == "__main__":
    run_detect()
