# framework/core/threads.py

import socket
import json
import time
import struct
import cv2
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal

# 引用现有模块
from framework.module import HostControl
from framework.module.transfer import DataProtocol

class DataReceiverThread(QThread):
    """数据接收线程：监听 4097 端口，保存图像"""
    data_received = pyqtSignal(int, bool)  # (pos_id, is_last)
    log_message = pyqtSignal(str)

    def __init__(self, output_root: Path, port=4097):
        super().__init__()
        self.output_root = output_root
        self.port = port
        self.is_running = True
        self.server_socket = None

    def run(self):
        HOST = '0.0.0.0'
        self.output_root.mkdir(exist_ok=True, parents=True)
        
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((HOST, self.port))
            self.server_socket.listen(1)
            self.log_message.emit(f"[Receiver] 监听端口 {self.port}...")

            while self.is_running:
                self.server_socket.settimeout(1.0)
                try:
                    conn, addr = self.server_socket.accept()
                except socket.timeout:
                    continue
                except Exception as e:
                    self.log_message.emit(f"[Receiver] Accept Error: {e}")
                    continue

                self.log_message.emit(f"[Receiver] 连接来自: {addr}")
                conn.settimeout(None)

                try:
                    while self.is_running:
                        try:
                            image, meta_data = DataProtocol.unpack_data(conn)
                        except (ConnectionError, struct.error):
                            self.log_message.emit("[Receiver] 连接断开")
                            break

                        if isinstance(meta_data, list): continue

                        pos_id = meta_data.get("pos_id", 0)
                        filename = meta_data.get("filename", f"unknown_{time.time()}.png")
                        is_last = meta_data.get("is_last", False)
                        defects = meta_data.get("defects", [])

                        pos_dir = self.output_root / f"pos{pos_id}"
                        pos_dir.mkdir(parents=True, exist_ok=True)
                        
                        cv2.imwrite(str(pos_dir / filename), image)

                        if defects:
                            with open(pos_dir / "defects.json", 'w', encoding='utf-8') as f:
                                json.dump(defects, f, indent=2, ensure_ascii=False)

                        self.data_received.emit(pos_id, is_last)
                finally:
                    conn.close()
        except Exception as e:
            self.log_message.emit(f"[Receiver] Error: {e}")
        finally:
            if self.server_socket: self.server_socket.close()

    def stop(self):
        self.is_running = False
        self.wait()


class RobotSimulationThread(QThread):
    """机器人/主机控制模拟线程"""
    log_message = pyqtSignal(str)
    task_finished = pyqtSignal()

    def __init__(self, project_folder: str):
        super().__init__()
        self.project_folder = project_folder
        self.host_control = None
        self.is_running = False

    def run(self):
        self.is_running = True
        try:
            self.log_message.emit("[Control] 初始化 HostControl...")
            self.host_control = HostControl(self.project_folder)
            
            self.log_message.emit("[Control] 等待从机连接 Port 4096...")
            self.host_control.Socket_init() 
            self.log_message.emit("[Control] 已连接！初始化投影...")
            self.host_control.Projected_init()

            target_points = [1, 2, 3] # 模拟3个点位
            for p in target_points:
                if not self.is_running: break

                self.log_message.emit(f"[Robot] >>> 移动到点位 {p} ...")
                time.sleep(2) 

                self.log_message.emit(f"[Robot] 到位 {p}，触发视觉拍照...")
                self.host_control.Take_photo() 
                self.log_message.emit(f"[Robot] 点位 {p} 完成")
                time.sleep(1) 

            self.log_message.emit("[Control] 任务全部完成")

        except Exception as e:
            self.log_message.emit(f"[Control] 异常: {e}")
        finally:
            if self.host_control:
                self.host_control.Disconnect()
            self.task_finished.emit()

    def stop(self):
        self.is_running = False