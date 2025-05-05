# -*-coding:utf-8 -*-
import os
import time
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
framework_root = current_file.parent.parent
sys.path.insert(0, str(framework_root))

from module import SocketServer, CameraControl, Host
import argparse

CAMERA_FLAG = 0     # 当前采集点位的数量
    
def write_camera_flag():
    #存储完毕CAMERA_FLAG+1
    global CAMERA_FLAG
    CAMERA_FLAG += 1
    print('CAMERA_FLAG:',CAMERA_FLAG)
    
    file_path = os.path.join(os.path.dirname(__file__),'FLAG.txt')           # txt文件路径         
    with open(file_path, 'w') as file:
        file.write(f'CAMERA_FLAG={CAMERA_FLAG}\n')              # txt写入文件

    print(f'CAMERA_FLAG={CAMERA_FLAG} 已写入 {file_path}')     

class CaptureTracker:
    """相机采集状态跟踪器"""
    def __init__(self):
        self.expected_serials = set()  # 预期需要接收的相机序列号
        self.received_serials = set()  # 实际接收到的序列号

def create_camera_with_callback(server, exposure=8000, max_frames=10):
    """创建带回调的相机工厂函数"""
    tracker = CaptureTracker()
    
    def _capture_callback(serial):
        """闭包函数捕获tracker和server"""
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
    """主控制逻辑封装"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--delay', type=int, default=1000)
    args = parser.parse_args()

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
                    camera.start_grabbing()

        except (RuntimeError, ConnectionError, KeyboardInterrupt) as e:  
            print(f"❌ 连接异常: {str(e)}")
            print("🕒 等待重新连接...")
            time.sleep(1)

if __name__ == '__main__':
    main()
