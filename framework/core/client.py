"""
文件: client.py
功能: 从机端核心控制模块，处理主机指令并同步相机采集
依赖: 
- module.SocketServer: 网络通信模块
- module.CameraControl: 相机控制模块
- argparse: 命令行参数解析

典型用法:
>>> python client.py --delay=1500  # 启动从机服务，设置重连延迟1.5秒
"""
# -*-coding:utf-8 -*-
import time
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
framework_root = current_file.parent.parent
sys.path.insert(0, str(framework_root))

from module import SocketServer, CameraControl
import argparse
from framework.engine.main_api import PipelineExecutor

class CaptureTracker:
    """多相机采集状态跟踪器
    
    主要职责:
    - 记录预期需要采集的相机序列号集合
    - 跟踪实际已完成的采集任务
    - 管理相机与主机的状态同步
    
    属性:
        expected_serials (set): 预期采集的相机序列号集合
        received_serials (set): 已接收的相机序列号集合
    """
    def __init__(self):
        self.expected_serials = set()  # 预期需要接收的相机序列号
        self.received_serials = set()  # 实际接收到的序列号

def create_camera_with_callback(server, exposure=8000, max_frames=10):
    """创建带协议状态管理的相机实例
    
    🌟 核心流程说明:
    1. 创建状态跟踪器 (CaptureTracker)
    2. 定义闭包回调函数，捕获tracker和server上下文
    3. 当所有相机完成采集时: 
       - 发送'switch_pattern'通知主机
       - 等待下个'capture_order'指令
    4. 返回绑定协议的相机控制器
    
    Args:
        server (SocketServer): 已连接的Socket服务实例
        exposure (int): 相机曝光时间（微秒），默认8000μs
        max_frames (int): 单次采集最大帧数，默认10帧
        
    Returns:
        tuple: (CameraControl实例, CaptureTracker状态跟踪器)
    """
    tracker = CaptureTracker()
    
    def _capture_callback(serial):
        """相机采集完成回调（闭包捕获tracker和server）
        
        协议流程:
        1. 添加当前相机序列号到received_serials
        2. 当所有expected_serials完成采集:
           a. 发送'switch_pattern'通知主机切换图案
           b. 清空接收状态
           c. 等待下个'capture_order'指令
        """
        tracker.received_serials.add(serial)
        
        # 当所有预期相机完成采集时
        if tracker.received_serials == tracker.expected_serials:    # 两个相机都拍摄完成
            server.safe_send("switch_pattern")
            tracker.received_serials.clear()  # 重置状态
            # 等待主机发送 capture_order 命令
            while True:
                next_order = server.safe_receive()
                if next_order != "capture_order":
                    raise RuntimeError(f"协议错误，期望 'capture_order'，收到 '{next_order}'")
                print(f"收到主机拍照指令 {next_order}")
                break            
    
    return CameraControl (
        exposure_time=exposure,
        max_frames=max_frames,
        capture_callback=_capture_callback
    ), tracker

def main():
    """从机端主控制循环
    
    🔄 状态机流程:
    1. 初始化Socket服务 (自动重试端口)
    2. 创建带协议绑定的相机控制器
    3. 等待主机'capture_order'指令
    4. 设置预期采集的相机集合
    5. 启动相机采集流程
    6. 采集时序控制（由--delay参数控制）
    
    命令行参数:
        --delay (int): 重连等待时间（毫秒），默认1000ms
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--delay', type=int, default=1000)
    args = parser.parse_args()
    executor = PipelineExecutor()       # 引入主流程处理函数，同时初始化检测模型等
    while True:
        try:                 
            with SocketServer(host='10.18.18.11', base_port=4096, retries=5) as server:
                camera, tracker = create_camera_with_callback(server)

                while True:
                    host_order = server.safe_receive()  
                    if host_order != "capture_order":
                        raise RuntimeError(f"协议错误，期望 'capture_order'，收到 '{host_order}'")
                    print(f'收到主机拍照指令 {host_order}')

                    tracker.expected_serials = set(camera.img_buffers.keys())
                    raw_imgs = camera.start_grabbing()
                    executor.execute_pipeline(raw_imgs)     # 执行整个算法处理流程

        except (RuntimeError, ConnectionError, KeyboardInterrupt) as e:  
            print(f"❌ 连接异常: {str(e)}")
            print("🕒 等待重新连接...")
            time.sleep(1)

if __name__ == '__main__':
    main()
