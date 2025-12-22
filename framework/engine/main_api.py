from pathlib import Path
import socket
import json
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import cv2  
import numpy as np
from framework.module.transfer import DataProtocol
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

        self.host_ip = '10.18.18.10' # 主机 IP (请根据实际情况修改)
        self.data_port = 4097        # 数据专用端口
        self.data_socket = None
        self._connect_host_blocking()

    def _connect_host_blocking(self):
        """阻塞式连接：直到连接成功才返回"""
        print(f"🔄 [Data] 正在连接主机数据服务 {self.host_ip}:{self.data_port}...")
        
        while self.data_socket is None:
            try:
                temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # 设置较短的超时，方便快速轮询
                temp_socket.settimeout(2.0) 
                temp_socket.connect((self.host_ip, self.data_port))
                
                # 连接成功
                temp_socket.settimeout(None) # 恢复阻塞模式
                self.data_socket = temp_socket
                print(f"✅ [Data] 初始化连接成功！数据通道已建立。")
                
            except (ConnectionRefusedError, TimeoutError, socket.timeout):
                print(f"⏳ [Data] 等待主机启动接收服务 (Port {self.data_port})...")
                time.sleep(1) # 等待 1 秒再重试
            except Exception as e:
                print(f"❌ [Data] 连接发生未知错误: {e}")
                time.sleep(1)

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
        import pdb
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

    # 请替换原有的 _save_results 方法
    def _save_results(self, result):
        """保存并发送所有结果（循环发送模式）"""
        pos_dir = self._create_pos_dir()
        current_pos_id = int(pos_dir.name.replace("pos", "")) # 获取当前点位ID (例如 1)
        
        # === 1. 准备要保存/发送的所有图像任务 ===
        # 格式: (image_data, filename)
        tasks = []

        # (1) 原始图像
        for cam_idx, cam_imgs in enumerate(result["raw_imgs"], start=1):
            for img, name in zip(cam_imgs, self.cfg.config['pmd']['image_names']):
                filename = f"cam{cam_idx}_{name}.png"
                tasks.append((img, filename))
                # 本地保存 (备份)
                cv2.imwrite(str(pos_dir / filename), img)

        # (2) 预处理图像
        for i, processed_img in enumerate(result["processed_img"], start=1):
            filename = f"processed_{i}.png"
            tasks.append((processed_img, filename))
            cv2.imwrite(str(pos_dir / filename), processed_img)
        
        # (3) 相位图 (需要从 Tensor 转 numpy)
        for i, phase_img in enumerate(result["abs_phase"], start=1):
            filename = f"phase_{i}.png"
            # Tensor -> Numpy -> Normalize(0-255) -> Color
            img_np = phase_img.cpu().numpy()
            img_norm = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
            
            tasks.append((img_color, filename))
            # 本地保存 (注意本地保存 float 原图还是 uint8 可视化图？这里保持原逻辑存 float)
            # 如果原逻辑是存 float，这里仅为了传输转了 uint8。
            # 这里为了简单，假设本地也存 uint8 预览，或者你可以保留原有的 cv2.imwrite 逻辑
            cv2.imwrite(str(pos_dir / filename), img_np) # 保持原有逻辑存 float/raw

        # (4) 准备缺陷数据
        defects_tensor = result["defects"]
        defects_list = [d.tolist() for d in defects_tensor]
        # 本地保存 JSON
        with open(pos_dir / "defects.json", "w") as f:
            json.dump(defects_list, f, indent=2)

        print(f"本地结果已保存至 {pos_dir}")

        if self.data_socket is None:
            # 只有极少数运行中途断线的情况会走到这里
            print("⚠️ [Data] 运行时连接丢失，尝试重连...")
            self._connect_host_blocking() # 再次进入阻塞重连

        # === 2. 循环发送数据 ===
        if self.data_socket:
            print(f"📤 [Data] 开始传输点位 {current_pos_id} 的数据，共 {len(tasks)} 张图像...")
            try:
                total = len(tasks)
                for i, (img, fname) in enumerate(tasks):
                    is_last = (i == total - 1)
                    
                    # 构造元数据 (Metadata)
                    # 只有最后一张图才附带 defects 数据，其他时候为空列表，节省带宽
                    meta_data = {
                        "pos_id": current_pos_id,
                        "filename": fname,
                        "is_last": is_last,
                        "defects": defects_list if is_last else [] 
                    }
                    
                    # 确保图像是 uint8 BGR 格式 (传输通用格式)
                    # 注意：如果 tasks 里的 img 已经是 uint8 BGR 则直接用，否则需要转换
                    # 上面 raw 和 processed 都是 BGR uint8，但 phase 需要注意
                    if img.dtype != np.uint8:
                         # 再次确保转换（防止上面 phase 本地存的是 float）
                         img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                         if len(img.shape) == 2:
                             img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                    # 打包发送
                    packet = DataProtocol.pack_data(img, meta_data)
                    self.data_socket.sendall(packet)
                    
                    # 可选：打印进度
                    # print(f"  -> 发送 {fname} ({i+1}/{total})")

                print(f"✅ [Data] 点位 {current_pos_id} 传输完成")
                
            except Exception as e:
                print(f"❌ [Data] 传输中断: {e}")
                self.data_socket.close()
                self.data_socket = None

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
