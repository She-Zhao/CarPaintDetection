from typing import List, Union
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import  torch.nn.functional as F

class Detectprocessor:
    """检测处理器"""
    def __init__(self, **kwargs):
        self.iou_thres = kwargs['iou_thres']
        self.conf_thres = kwargs['conf_thres']
        self.stride = kwargs['stride']
        self.imgsz = kwargs['imgsz']
        self.selected_model = kwargs['selected_model']
        self.model_path = kwargs['all_models'][self.selected_model]
        self.multi_img = True if self.selected_model == 'MPFF' else False
        self.model = self.init_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def init_model(self) -> Union[YOLO]:
        """初始化模型

        Returns:
            YOLO: 检测模型
        """
        # 加载Pytorch模型
        weights_path = Path(__file__).parent / "weights" / self.model_path      # 从framework开始加载
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
        abs_phase: List[torch.Tensor],
        processed_imgs: List[np.ndarray]
    ) -> List[torch.Tensor]:
        """返回检测结果

        Args:
            abs_phase (List[torch.Tensor]): 绝对相位图, 内层图像维度需要为二维的(H, W)
            processed_imgs (List[np.ndarray]): 前处理的拼接结果, 内层图像维度需要为二维的(H, W)

        Returns:
            List[torch.Tensor]: 返回的Tensor列表, 每个Tensor的数据类型为torch.float32, 形状为 [class, x, y, w, h, conf]
        
        Notes:
            YOLO的输入对tensor和ndarry的支持情况不同:
            输入tensor时, 需要进行resize、归一化、维度调整到(B, C, H, W)
            输入ndarray时, 保证维度是(H, W, 3)即可, 剩下的YOLO内部完成。        
        """
        images = self._preprocess(abs_phase=abs_phase, processed_imgs=processed_imgs)

        results = self.model(images, iou=self.iou_thres, conf=self.conf_thres, device=self.device, imgsz=[2048,2464], multi_img=self.multi_img)

        # 后处理返回List[torch.Tensor]，Tensor:[[c, x, y, w, h, conf], [], ...]
        return [
            torch.cat([r.boxes.cls.unsqueeze(1), r.boxes.xywh, r.boxes.conf.unsqueeze(1)], dim=1)
            for r in results
        ]
    
    def _preprocess(self, abs_phase, processed_imgs):
        if self.selected_model == 'PMD':
            return self._preprocess_pmd(abs_phase) 
        elif self.selected_model == 'MPFF':
            return self._preprocess_mpff(abs_phase, processed_imgs)
        else:
            print(
                f"选择的模型只能是 `PMD` 或者 `MPFF` 其中之一！"
                f"当前选择的模型 {self.selected_model} 不符合要求, 将使用 `PMD`"
            )
            return self._preprocess_pmd(abs_phase) 
    
    def _preprocess_pmd(self, abs_phase: List[torch.Tensor]) -> torch.Tensor:
        """对PMD模型进行前处理

        Args:
            abs_phase (List[torch.Tensor]): 绝对相位图

        Returns:
            torch.Tensor: 模型要求的输入图像
        """
        # 转换为(B, ?, H, W)的Tensor
        if not isinstance(abs_phase, List) or not isinstance(abs_phase[0], torch.Tensor):
            raise TypeError(
                f"输入的图像不符合格式要求！"
                f"要求绝对相位图类型：{List[torch.Tensor]}"
                f"当前绝对相位图类型: {type(abs_phase)}, dtpye={type(abs_phase[0])}"                  
            )
        x = torch.stack(abs_phase, dim=0)                   # List[(H, W)] -> torch.Tensor(B, H, W)
        if x.ndim == 3: x = x.unsqueeze(1)                  # (B, H, W) -> (B, 1, H, W)
        
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

    def _preprocess_mpff(self, abs_phase:List[torch.Tensor], processed_imgs:List[np.ndarray]) -> List[torch.Tensor]:
        """mpff模型的前处理函数

        Args:
            abs_phase (List[torch.Tensor]): 绝对相位图
            processed_imgs (List[np.ndarray]): 拼接后的图像

        Returns:
            List[torch.Tensor]: mpff模型的输入。五个通道依次是: [AP, sin0, sin1, sin2, sin3]
        """

        if not isinstance(processed_imgs, List) or not isinstance(abs_phase, List):
            raise TypeError(
                f"输入的图像不符合格式要求！"
                f"要求绝对相位图类型：{List[torch.Tensor]}, 正弦图类型：{List[np.ndarray]}"
                f"当前绝对相位图类型: {type(abs_phase)}, 正弦图类型: {type(processed_imgs)}"               
            )
        
        sines_tensor = torch.from_numpy(np.stack(processed_imgs[-4:], axis=0))               
        sines_tensor = sines_tensor.to(self.device, dtype=torch.float32)                    # (4, H, W)
        phase_tensor = torch.stack(abs_phase, dim=0).to(self.device, dtpye=torch.float32)   # (1, H, W)

        mpff_input = torch.cat([phase_tensor, sines_tensor], dim=0).unsqueeze(0)      # (1, 5, H, W)
        
        _, _, h, w = mpff_input.shape
        right_pad = (self.stride - w%self.stride) % self.stride
        bottom_pad = (self.stride - h%self.stride) % self.stride
        
        if right_pad>0 or bottom_pad>0:
            mpff_input = F.pad(mpff_input, (0, right_pad, 0, bottom_pad), mode='constant', value=114)
            
        return mpff_input / 255.0
