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
import argparse
from framework.module import SocketServer, CameraControl
from framework.engine.main_api import PipelineExecutor
from framework.module.model_config import ModelConfigManager

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
    
    return CameraControl(
        exposure_time=exposure,
        max_frames=max_frames,
        capture_callback=_capture_callback
    ), tracker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delay', type=int, default=1000)
    args = parser.parse_args()
    
    config = ModelConfigManager()
    executor = PipelineExecutor(config)

    while True:
        camera = None  # [关键] 循环开始前重置变量
        try:                 
            with SocketServer(host='10.18.18.11', base_port=4096, retries=5) as server:
                # 创建新相机对象
                camera, tracker = create_camera_with_callback(server)

                while True:
                    host_order = server.safe_receive()  
                    if host_order != "capture_order":
                        # 收到异常指令，跳出内部循环，触发资源清理
                        raise RuntimeError(f"协议错误: {host_order}")
                    
                    print(f'收到主机拍照指令 {host_order}')

                    tracker.expected_serials = set(camera.img_buffers.keys())
                    raw_imgs = camera.start_grabbing()
                    executor.execute_pipeline(raw_imgs)

        except (RuntimeError, ConnectionError, KeyboardInterrupt, Exception) as e:  
            # [修改] 捕获所有 Exception，防止 Pylon 报错直接崩掉程序
            print(f"❌ 运行异常: {str(e)}")
            print("🕒 准备重新连接...")
            time.sleep(1)
            
        finally:
            # [关键修改] 无论是因为断网、报错还是正常结束，
            # 只要 camera 对象存在，就强制关闭它，释放硬件锁。
            if camera:
                print("正在清理相机资源...")
                camera.release()
                camera = None # 防止重复释放

if __name__ == '__main__':
    main()