from pathlib import Path
import socket
import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from framework.module.transfer import DataProtocol
import cv2  
import numpy as np
from framework.engine.preprocess.preprocess_api import Preprocessor
from framework.engine.pmd.pmd_api import PMDprocessor
from framework.engine.detect.detect_api import Detectprocessor
from framework.module import ModelConfigManager

class PipelineExecutor:
    """算法pipeline执行器
    异步执行采用ThreadPoolExecutor, 因为ThreadPoolExecutor 内置"异步执行+自动队列​"​, 可以省略很多麻烦
    异步执行含义：前面拍完照，调用PipelineExecutor，会将任务放在单独一个线程执行，然后立刻切换到主线程进行下一个任务
    自动队列：即使前面没处理完，来了新照片，会自动进入缓存队列，当前面执行完了再执行当前任务
    """
    # 说明：这里
    def __init__(self, config: ModelConfigManager):
        """加载预处理、二值化的相关参数，初始化检测模型

        Args:
            config (ModelConfigManager): 相关参数和模型的配置
        """
        self.executor = ThreadPoolExecutor(max_workers=1)    # 不是为了并行，而是为了新任务会自动排队等待，严格保持先进先出（FIFO）顺序
        self.output_root = Path(__file__).parent.parent.parent / "output"      # CarPaintDetection/output
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.cfg = config
        self._init_algorithm_modules()
        self.pos_idx = 0

        self.host_ip = '10.18.18.1' # 主机 IP (请根据实际情况修改)
        self.data_port = 4097        # 数据专用端口
        self.data_socket = None
        self._try_connect_host()

    # === [新增] 连接辅助函数 ===
    def _try_connect_host(self):
        """尝试连接主机，连接失败不报错，只打印警告"""
        try:
            if self.data_socket:
                self.data_socket.close()
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.data_socket.settimeout(1.0) # 1秒连接超时
            self.data_socket.connect((self.host_ip, self.data_port))
            self.data_socket.settimeout(None) # 连接后取消超时（或设长一点）
            print(f"✅ [Data] 已连接到主机数据服务 {self.host_ip}:{self.data_port}")
        except Exception as e:
            print(f"⚠️ [Data] 连接主机失败: {e} (将仅保存本地)")
            self.data_socket = None

    def _init_algorithm_modules(self):
        """动态加载算法模块（保持扩展性）"""
        self.preprocess = Preprocessor()
        self.pmd = PMDprocessor()
        self.detect = Detectprocessor(**self.cfg.config['detect'])

    def execute_pipeline(self, raw_imgs:List[List[np.ndarray]], debug=False):
        """算法执行管线的外部接口
        
        Args:
            raw_imgs：两组相机采集的原始图像
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

    def _execute_pipeline(self, raw_imgs: List[List[np.ndarray]]) -> Dict:           # List[List[np.ndarray]]
        """算法执行pipeline"""
        # 1. 图像预处理（拼接、有效区域提取等）
        processed_imgs = self.preprocess(raw_imgs, self.cfg.param['H_matrix'])      # processed_imgs: List[np.ndarray],图像为(H, W)二维
        
        # 2. PMD相位计算
        abs_phase = self.pmd(processed_imgs, self.cfg.param['th_dict'][f'pos{self.pos_idx}'])           # abs_phase: [GPU.tensor, GPU.tensor]
        
        # 3. 缺陷检测
        defects = self.detect(abs_phase=abs_phase, processed_imgs=processed_imgs)   # defects: [GPU.tensor(n1,6), GPU.tensor(n2,6)] -> [c,x,y,w,h,conf]
         
        # 更新当前点位
        self.pos_idx += 1
        return {
            "raw_imgs": raw_imgs,                       # 原始图像
            "processed_img": processed_imgs,            # 预处理结果
            "abs_phase": abs_phase,                    # 相位计算结果
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
            for img, name in zip(cam_imgs, self.cfg.config['pmd']['image_names']):
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
        defects_tensor = result["defects"]
        defects_list = [defect.tolist() for defect in defects_tensor]
        with open(pos_dir / "defects.json", "w") as f:
            json.dump(defects_list, f, indent=2)
        
        print(f"结果已保存至 {pos_dir}")
        
        # === [新增] 发送数据到主机 ===
        if self.data_socket:
            try:
                print("📤 [Data] 正在向主机发送数据...")
                
                # 1. 准备图片 (这里以发送第一张相位图为例，你也可以改发拼接图)
                # result["abs_phase"] 是 List[Tensor]
                phase_tensor = result["abs_phase"][0] 
                img_data = phase_tensor.cpu().numpy()
                
                # 归一化转为 uint8 图片格式 (0-255)
                img_data = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR) # 转3通道
                
                # 2. 准备 JSON 数据
                defects_tensor = result["defects"]
                defects_list = [d.tolist() for d in defects_tensor]
                
                # 3. 打包并发送
                packet = DataProtocol.pack_data(img_data, defects_list)
                self.data_socket.sendall(packet)
                print(f"✅ [Data] 发送成功 ({len(packet)} bytes)")
                
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                print("❌ [Data] 连接断开，尝试重连...")
                self._try_connect_host()
            except Exception as e:
                print(f"❌ [Data] 发送异常: {e}")

    def _create_pos_dir(self) -> Path:
        """创建递增的pos目录"""
        # 获取所有以pos开头的目录名中的数字部分
        pos_numbers = [
            int(d.name[3:]) for d in self.output_root.glob("pos*")
            if d.is_dir() and d.name[3:].isdigit()
        ]
        
        new_num = max(pos_numbers, default=0) + 1
        new_dir = self.output_root / f"pos{new_num}"
        new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir
