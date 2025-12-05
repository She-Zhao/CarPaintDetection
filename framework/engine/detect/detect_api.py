from typing import List, Union
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import  torch.nn.functional as F

class Detectprocessor:
    """检测处理器"""
    def __init__(self,
                 iou_thres: float = 0.5,
                 conf_thres: float = 0.25,
                 stride: int = 32):
        self.iou_thres = iou_thres
        
        self.conf_thres = conf_thres
        self.stride = stride
        self.model = self.init_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def init_model(self) -> YOLO:
        """初始化模型

        Returns:
            YOLO: 检测模型
        """
        # 加载Pytorch模型
        weights_path = Path(__file__).parent / "weights" / "yolo11s_pmd847.pt"
        model = YOLO(weights_path)
        return model

        # 初始化TensorRT引擎模型
        # 当在 YOLO(...) 初始化时传入 .engine 文件路径，Ultralytics 库会自动识别并在底层调用 TensorRT 的 Python 绑定来进行推理。
        # engine_path = Path(__file__).parent / "weights" / "yolo11s_pmd847_fp32.engine"
        # assert engine_path.exists(), f"Engine file missing at {engine_path}"
        # return YOLO(engine_path)

    # 对外暴漏的接口
    def __call__(
        self, 
        inputs: Union[torch.Tensor, np.ndarray, List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """返回检测结果

        Args:
            inputs: 输入待检测图像，支持二/三维的Tensor/ndarray、List[Tensor](Tensor必须是二维)
        
        Returns:
            List[torch.Tensor]: 返回的Tensor列表
            每个Tensor的数据类型为torch.float32
            形状为 [class, x, y, w, h, conf]
        
        Notes:
            YOLO的输入对tensor和ndarry的支持情况不同:
            输入tensor时, 需要进行resize、归一化、维度调整到(B, C, H, W)
            输入ndarray时， 保证维度是(H, W, 3)即可, 剩下的YOLO内部完成
        """
        images = self._preprocess(inputs)

        results = self.model(images, iou=self.iou_thres, conf=self.conf_thres, device=self.device, imgsz=[2048,2464])

        # 后处理返回List[torch.Tensor]，Tensor:[[c, x, y, w, h, conf], [], ...]
        return [
            torch.cat([r.boxes.cls.unsqueeze(1), r.boxes.xywh, r.boxes.conf.unsqueeze(1)], dim=1)
            for r in results
        ]

    def _preprocess(self, inputs):
        """对图像依次进行格式转换(B, ?, H, W的GPU Tensor)、填充、维度变换(B, 3, H, W)
        输入需要是二通道或者三通道，三通道必须是(C, H, W)
        """
        # 转换为(B, ?, H, W)的Tensor
        if isinstance(inputs, List):
            x = torch.stack(inputs, dim=0)
            if x.ndim == 3: x = x.unsqueeze(1)                  # (B, H, W) -> (B, 1, H, W)
        
        elif isinstance(inputs, torch.Tensor):
            x = inputs
            if x.ndim == 2: x = x.unsqueeze(0).unsqueeze(0)     # (H, W) -> (1, 1, H, W)
            elif x.ndim == 3: x = x.unsqueeze(0)                # (C, H, W) -> (1, C, H, W)
        
        elif isinstance(inputs, np.ndarray):
            x = torch.as_tensor(inputs)
            if x.ndim == 2: x = x.unsqueeze(0).unsqueeze(0)     # (H, W) -> (1, 1, H, W)
            elif x.ndim == 3: x = x.unsqueeze(0)                # (C, H, W) -> (1, C, H, W)
        
        # 填充
        _, _, h, w = x.shape
        right_pad = (self.stride - w%self.stride) % self.stride     # 计算填充的尺寸
        bottom_pad = (self.stride - h%self.stride) % self.stride
        if right_pad>0 or bottom_pad>0:
            x = F.pad(x, pad=(0, right_pad, 0, bottom_pad), mode='constant', value=114)
        
        # 转换为(B, 3, H, W)的GPU Tensor，并进行数据类型的转换和归一化
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1).to(device=self.device, dtype=torch.float32) / 255.0
        
        return x
