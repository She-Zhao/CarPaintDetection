# framework/engine/main_api.py

from pathlib import Path
import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import cv2  
import numpy as np

# 引入新模块
from framework.module.data_sender import DataSender
from framework.module import ModelConfigManager

# 算法模块
from framework.engine.preprocess.preprocess_api import Preprocessor
from framework.engine.pmd.pmd_api import PMDprocessor
from framework.engine.detect.detect_api import Detectprocessor

class PipelineExecutor:
    """算法pipeline执行器
    异步执行采用ThreadPoolExecutor, 因为ThreadPoolExecutor 内置"异步执行+自动队列​"​, 可以省略很多麻烦
    异步执行含义：前面拍完照，调用PipelineExecutor，会将任务放在单独一个线程执行，然后立刻切换到主线程进行下一个任务
    自动队列：即使前面没处理完，来了新照片，会自动进入缓存队列，当前面执行完了再执行当前任务
    """
    
    def __init__(self, config: ModelConfigManager):
        self.cfg = config
        self.pos_idx = 0
        
        # 1. 基础资源初始化
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.output_root = Path(__file__).parent.parent.parent / "output"
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 2. 算法模块初始化
        self.preprocess = Preprocessor()
        self.pmd = PMDprocessor()
        self.detect = Detectprocessor(**self.cfg.config['detect'])

        # 3. 网络发送器初始化 (逻辑被移交出去了，这里很清爽)
        self.sender = DataSender(host_ip='10.18.18.10', port=4097)

    def execute_pipeline(self, raw_imgs:List[List[np.ndarray]], debug=False):
        """外部调用接口"""
        if debug:
            result = self._execute_pipeline(raw_imgs)
            self._handle_results(result)
        else:
            future = self.executor.submit(self._execute_pipeline, raw_imgs)
            future.add_done_callback(self._result_callback)

    def _execute_pipeline(self, raw_imgs) -> Dict:
        """核心算法流程"""
        # 1. 预处理
        processed_imgs = self.preprocess(raw_imgs, self.cfg.param['H_matrix'])
        # 2. 相位计算
        abs_phase = self.pmd(processed_imgs, self.cfg.param['th_dict'][f'pos{self.pos_idx}'])
        # 3. 缺陷检测
        defects = self.detect(abs_phase=abs_phase, processed_imgs=processed_imgs)
        
        self.pos_idx += 1
        return {
            "raw_imgs": raw_imgs,
            "processed_img": processed_imgs,
            "abs_phase": abs_phase,
            "defects": defects,
        }

    def _result_callback(self, future):
        try:
            self._handle_results(future.result())
        except Exception as e:
            print(f"处理结果回调异常: {e}")

    def _handle_results(self, result):
        """处理结果：本地保存 + 网络发送"""
        # 1. 准备数据并保存到本地
        pos_dir, tasks, defects_list = self._save_locally(result)
        
        # 2. 发送到主机 (委托给 Sender)
        current_pos_id = int(pos_dir.name.replace("pos", ""))
        self.sender.send_batch(tasks, current_pos_id, defects_list)

    def _save_locally(self, result):
        """辅助函数：处理本地保存逻辑，并返回待发送的任务列表"""
        pos_dir = self._create_pos_dir()
        tasks = [] # [(img, filename), ...]

        # (1) 保存原始图
        for cam_idx, cam_imgs in enumerate(result["raw_imgs"], start=1):
            for img, name in zip(cam_imgs, self.cfg.config['pmd']['image_names']):
                filename = f"cam{cam_idx}_{name}.png"
                tasks.append((img, filename))
                cv2.imwrite(str(pos_dir / filename), img)

        # (2) 保存预处理图
        for i, processed_img in enumerate(result["processed_img"], start=1):
            filename = f"processed_{i}.png"
            tasks.append((processed_img, filename))
            cv2.imwrite(str(pos_dir / filename), processed_img)
        
        # (3) 保存相位图
        for i, phase_img in enumerate(result["abs_phase"], start=1):
            filename = f"phase_{i}.png"
            img_np = phase_img.cpu().numpy() # Tensor -> Numpy
            
            # 存本地可以用原始float数据，也可以转uint8，这里保持跟你之前逻辑一致
            # 如果需要发送，Sender会自动转uint8，这里我们只需负责存盘
            cv2.imwrite(str(pos_dir / filename), img_np)
            
            # 注意：发送列表里最好放转换好的 uint8 彩色图，或者让 Sender 去转
            # 这里为了简单，我们传原始 numpy 给 Sender，Sender 里有转换逻辑
            tasks.append((img_np, filename))

        # (4) 保存缺陷数据
        defects_list = [d.tolist() for d in result["defects"]]
        with open(pos_dir / "defects.json", "w") as f:
            json.dump(defects_list, f, indent=2)

        print(f"本地保存完成: {pos_dir}")
        return pos_dir, tasks, defects_list

    def _create_pos_dir(self) -> Path:
        """创建递增的pos目录"""
        pos_numbers = [
            int(d.name[3:]) for d in self.output_root.glob("pos*")
            if d.is_dir() and d.name[3:].isdigit()
        ]
        new_num = max(pos_numbers, default=0) + 1
        new_dir = self.output_root / f"pos{new_num}"
        new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir