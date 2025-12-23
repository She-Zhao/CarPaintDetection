# framework/module/data_sender.py

import socket
import time
import cv2
import numpy as np
from framework.module.protocol import DataProtocol

class DataSender:
    """负责将检测结果发送到主机
    职责：
    1. 管理 Socket 连接（阻塞连接、自动重连）
    2. 封装数据包发送逻辑
    """
    def __init__(self, host_ip, port):
        self.host_ip = host_ip
        self.port = port
        self.socket = None
        
        # 初始化时尝试阻塞连接
        self._connect_blocking()

    def _connect_blocking(self):
        """阻塞式连接，直到成功"""
        print(f"🔄 [Sender] 正在连接主机 {self.host_ip}:{self.port}...")
        while self.socket is None:
            try:
                temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                temp_socket.settimeout(2.0)
                temp_socket.connect((self.host_ip, self.port))
                
                temp_socket.settimeout(None)
                self.socket = temp_socket
                print(f"✅ [Sender] 连接成功！通道已建立。")
            except (ConnectionRefusedError, TimeoutError, socket.timeout):
                print(f"⏳ [Sender] 等待主机启动接收服务...")
                time.sleep(1)
            except Exception as e:
                print(f"❌ [Sender] 连接错误: {e}")
                time.sleep(1)

    def send_batch(self, tasks: list, pos_id: int, defects_data: list):
        """发送一批图像数据
        Args:
            tasks: List[(image_data, filename)], 待发送的图像列表
            pos_id: 当前点位ID
            defects_data: 缺陷数据列表
        """
        # 1. 检查连接，如果断开则重连
        if self.socket is None:
            print("⚠️ [Sender] 连接断开，正在重连...")
            self._connect_blocking()

        if self.socket:
            print(f"📤 [Sender] 正在发送点位 {pos_id} 数据 ({len(tasks)} 张)...")
            try:
                total = len(tasks)
                for i, (img, fname) in enumerate(tasks):
                    is_last = (i == total - 1)
                    
                    # 构造元数据
                    meta_data = {
                        "pos_id": pos_id,
                        "filename": fname,
                        "is_last": is_last,
                        "defects": defects_data if is_last else []
                    }
                    
                    # 格式转换：确保是 uint8 BGR
                    if img.dtype != np.uint8:
                        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                    # 打包发送
                    packet = DataProtocol.pack_data(img, meta_data)
                    self.socket.sendall(packet)
                
                print(f"✅ [Sender] 点位 {pos_id} 发送完成")

            except Exception as e:
                print(f"❌ [Sender] 发送中断: {e}")
                self.close() # 标记断开，触发下次重连

    def close(self):
        """安全关闭连接"""
        if self.socket:
            try:
                self.socket.close()
            except: pass
        self.socket = None