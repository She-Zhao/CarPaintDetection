# engine/processor.py
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from importlib import import_module
import cv2
import os
# from preprocess.run import run_preprocess
# from pmd.run import run_pmd
# from yolo.run import run_detect

IMAGE_NAMES = ["gc0", "gc1", "gc2", "gc3", "gc4", "sin0", "sin1", "sin2", "sin3"]

class PipelineExecutor:
    # 说明：这里采用ThreadPoolExecutor，是因为ThreadPoolExecutor 内置"异步执行+自动队列​"​
    #  ​​max_workers 仅控制并行度
    # 异步执行含义：前面拍完照，调用PipelineExecutor，会将任务放在单独一个线程执行，然后立刻回到拍照中
    # 自动队列：即使前面没处理完，来了新照片，会自动进入缓存队列，当前面执行完了再执行当前任务
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)       # 重要！新任务会自动排队等待，严格保持先进先出（FIFO）顺序
        self.output_root = Path(__file__).parent.parent / "output"      # framework/output
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._init_algorithm_modules()
        self.model = import_module("framework.engine.detect.detect_api").init_model
        
    # def init_model(self):
    #     return model

    def _init_algorithm_modules(self):
        """动态加载算法模块（保持扩展性）"""
        self.preprocess = import_module("framework.engine.preprocess.preprocess_api").run_preprocess
        self.pmd = import_module("framework.engine.pmd.pmd_api").run_pmd
        self.detect = import_module("framework.engine.detect.detect_api").run_detect

    def execute_pipeline(self, raw_imgs, debug=False):
        """主入口：支持调试模式
        Args:
            debug: True时同步执行，False时异步执行
        """
        if debug:
            # 同步调试模式
            result = self._execute_pipeline(raw_imgs)
            self._save_results(result)  # 直接使用结果
        else:
            # 异步部署模式
            future = self.executor.submit(self._execute_pipeline, raw_imgs)
            future.add_done_callback(self._result_callback)

    def _execute_pipeline(self, raw_imgs):           # List[List[np.ndarray]]
        """顺序执行处理链"""
        # 1. 图像预处理（拼接、有效区域提取等）
        processed_imgs = self.preprocess(raw_imgs)      # processed_imgs:未知，豪杰还没做。预计np.ndarray
        
        # 2. PMD相位计算（假设输入为双图）
        abs_phases = self.pmd(processed_imgs)           # abs_phases: [GPU.tensor, GPU.tensor]
        
        # 3. 缺陷检测
        defects = self.detect(self.model, abs_phases)   # defects: [GPU.tensor(n1,6), GPU.tensor(n2,6)] -> [c,x,y,w,h,conf]
        
        return {
            "raw_imgs": raw_imgs,                       # 原始图像
            "processed_img": processed_imgs,            # 预处理结果
            "abs_phase": abs_phases,                    # 相位计算结果
            "defects": defects,                         # 缺陷检测结果
        }

    def _result_callback(self, future):
        """异步回调适配器"""
        try:
            self._save_results(future.result())
        except Exception as e:
            print(f"保存失败: {str(e)}")

    def _save_results(self, result):
        """统一保存结果（同步/异步共用）"""
        pos_dir = self._create_pos_dir()
        
        # 保存原始图像（文件名包含相机编号）
        for cam_idx, cam_imgs in enumerate(result["raw_imgs"], start=1):
            for img, name in zip(cam_imgs, IMAGE_NAMES):
                filename = f"cam{cam_idx}_{name}.png"
                path = pos_dir / filename
                cv2.imwrite(str(path), img)

        # 保存预处理结果
        for i, processed_img in enumerate(result["processed_img"], start=1):
            cv2.imwrite(str(pos_dir / f"processed_{i}.png"), processed_img)
        
        # 保存相位图
        for i, phase_img in enumerate(result["abs_phase"], start=1):
            cv2.imwrite(str(pos_dir / f"phase_{i}.png"), phase_img.cpu().numpy())
            
        # 保存检测结果
        with open(pos_dir / "defects.json", "w") as f:
            json.dump(result["defects"], f)
        
        print(f"结果已保存至 {pos_dir}")

    def _create_pos_dir(self) -> Path:
        """创建递增的pos目录（优化版）"""
        # 获取所有以pos开头的目录名中的数字部分
        pos_numbers = [
            int(d.name[3:]) for d in self.output_root.glob("pos*")
            if d.is_dir() and d.name[3:].isdigit()
        ]
        
        new_num = max(pos_numbers, default=0) + 1
        new_dir = self.output_root / f"pos{new_num}"
        new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir

if __name__ == "__main__":
    datapath1 = r'D:\Project\_New_System\test\pos1'
    datapath2 = r'D:\Project\_New_System\test\pos2'
    imgs1 = [cv2.imread(os.path.join(datapath1, img), cv2.IMREAD_GRAYSCALE) 
                for img in sorted(os.listdir(datapath1))]
    imgs2 = [cv2.imread(os.path.join(datapath2, img), cv2.IMREAD_GRAYSCALE) 
                for img in sorted(os.listdir(datapath2))] 
    
    # 测试用例
    executor = PipelineExecutor()
    test_images = [imgs1, imgs2]     # 替换为实际图像数据
    executor.execute_pipeline(test_images, debug=True)

