from typing import List, Union
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path


def init_model():
    # 在PipelineExecutor.__init__.py中调用，初始化模型加载
    weights_path = Path(__file__).parent / "weights" / "yolo11s_pmd847.pt"
    model = YOLO(weights_path)
    return model

# 对外暴漏的接口
def run_detect(model,
    abs_phases: List[torch.Tensor]
) -> List[List[float]]:
    """调用初始化好的模型，输出[x,y,w,h]这样一个bbox

    Args:
        imgs: 两张绝对相位图,存储在一个List中, 相位图Shape为(H,W)
    
    Returns:
        网络的检测结果，bbox的中心点坐标及宽和高
    """
    # 堆叠图像并转换为RGB格式（将单通道重复为3通道）并归一化
    images = torch.stack([
        phase.unsqueeze(0).repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
        for phase in abs_phases
    ], dim=0) / 255.0  # [2, 3, H, W] - 包含2张RGB图像的batch

    # 推理
    results = model(images, iou=0.5, conf=0.25, imgsz=images.shape[-2:], device="cuda")

    # 处理检测结果
    output = []
    for result in results:
        # 提取检测信息：类别、置信度和归一化边界框
        detections = torch.cat([
            result.boxes.cls.unsqueeze(1),    # 类别标签c
            result.boxes.xywhn,               # 归一化边界框 (x, y, w, h)
            result.boxes.conf.unsqueeze(1)    # 置信度分数conf
        ], dim=1)
        output.append(detections)
        
    return output
        
if __name__ == "__main__":
    run_detect()
