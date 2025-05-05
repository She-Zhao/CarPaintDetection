# -*-coding:utf-8 -*-
import numpy as np
import os
import pypylon.pylon as py
import cv2
import time

import socket
import argparse
from typing import Optional, Tuple

IMAGE_NAMES= ["zhj","gc0","gc1","gc2","gc3","gc4","sin0","sin1","sin2","sin3"]#存储图像的命名顺序
CAMERA_FLAG = 0     #当前采集点位的数量
CAMERA_NUM = 5      #多少个点位计算一次，需要从别的位置import，必须放在main外面
    
def write_camera_flag():
    #存储完毕CAMERA_FLAG+1
    global CAMERA_FLAG
    CAMERA_FLAG += 1
    print('CAMERA_FLAG:',CAMERA_FLAG)
    
    file_path = os.path.join(os.path.dirname(__file__),'FLAG.txt')           # txt文件路径         
    with open(file_path, 'w') as file:
        file.write(f'CAMERA_FLAG={CAMERA_FLAG}\n')              # txt写入文件

    print(f'CAMERA_FLAG={CAMERA_FLAG} 已写入 {file_path}')     


class SocketServer:
    """可自动重试的 TCP 服务端封装类"""
    def __init__(self, host: str, base_port: int, retries: int = 3):
        """
        Args:
            host: 监听地址
            base_port: 起始端口号
            retries: 端口冲突时的重试次数
        """
        self.host = host
        self.base_port = base_port
        self.retries = retries
        self.server_socket: Optional[socket.socket] = None
        self.connection: Optional[socket.socket] = None
        self.connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self) -> Tuple[socket.socket, socket.socket]:
        """建立 socket 连接"""
        current_port = self.base_port
        for attempt in range(self.retries):
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((self.host, current_port))
                self.server_socket.listen(10)
                print(f"✅ 成功监听 {self.host}:{current_port}")
                
                self.connection, address = self.server_socket.accept()
                print(f"🔗 已接受来自 {address} 的连接")
                self.connected = True
                return self.server_socket, self.connection
                
            except OSError as e:
                if attempt < self.retries - 1:
                    print(f"⚠️ 端口 {current_port} 被占用，尝试 {current_port+1}...")
                    current_port += 1
                else:
                    raise RuntimeError(f"无法绑定端口，最后尝试端口: {current_port}") from e

    def safe_receive(self, bufsize: int = 1024) -> str:
        """带连接状态检查的数据接收方法"""
        if not self.connected:
            raise ConnectionError("连接未建立")
            
        data = self.connection.recv(bufsize)
        if not data:
            self.connected = False
            raise ConnectionAbortedError("连接被主机关闭")
            
        return data.decode('utf-8')

    def safe_send(self, message: str) -> None:
        """带连接状态检查的数据发送方法"""
        if not self.connected:
            raise ConnectionError("连接未建立")
            
        self.connection.sendall(message.encode('utf-8'))

    def close(self) -> None:
        """安全关闭连接"""
        if self.connection:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
            except OSError:
                pass
        if self.server_socket:
            try:
                self.server_socket.close()
            except OSError:
                pass
        self.connected = False
        print("🔒 Socket 连接已安全关闭")

class Camera:
    def __init__(self, exposure_time=8000, height=None, width=None, 
                 max_frames=10, capture_callback=None):
        self.exposure_time = exposure_time
        self.height = height
        self.width = width        
        self.max_frames = max_frames
        self.capture_callback = capture_callback 
        self._camera_init()
        
    def _camera_init(self):
        try:
            tlf = py.TlFactory.GetInstance()
            di = py.DeviceInfo()
            di.SetDeviceClass("BaslerGigE") 
        
            devs = tlf.EnumerateDevices([di,])      # 发现可用设备
            if not devs:
                raise RuntimeError("未检测到任何Basler GigE相机")

            # 根据实际设备数量创建相机数组
            num_cameras = len(devs)
            print(f"发现 {num_cameras} 台相机")
            
            self.cam_array = py.InstantCameraArray(num_cameras)
            self.img_buffers = {}

            # 绑定并初始化相机
            for idx, cam in enumerate(self.cam_array):
            
                # 关联物理设备
                cam.Attach(tlf.CreateDevice(devs[idx]))
                cam.Open()
                
                # 配置基础参数
                if self.exposure_time: cam.ExposureTime.SetValue(self.exposure_time)
                if self.height: cam.Height.Value = self.height
                if self.width: cam.Width.Value = self.width
                
                # 生成唯一标识
                serial = int(cam.DeviceInfo.GetSerialNumber())
                cam.SetCameraContext(serial)            # 使用序列号作为上下文标识，传入的参数必须是int型
                self.img_buffers[serial] = []

        except py.GenericException as e:
            print(f"Pylon错误: {e}")
    
    def start_grabbing(self):
        try:
            self.cam_array.StartGrabbing(py.GrabStrategy_LatestImageOnly)
            # 两个相机从调用StartGrabbing到可以拍照需要时间，不加延时的话其中一个相机准备好了会先拍照导致时许对不上
            time.sleep(0.05)
            grab_timeout = 5000  # 超时时间5秒
            frame_counts = {serial:0 for serial in self.img_buffers.keys()}

            while True:
                # 获取图像结果
                res = self.cam_array.RetrieveResult(grab_timeout, py.TimeoutHandling_ThrowException)
                try:
                    if res.GrabSucceeded():
                        serial = res.GetCameraContext()
                        self.img_buffers[serial].append(res.Array)
                        print(f">>>>>相机{serial}，照片数量为{len(self.img_buffers[serial])}")

                        if self.capture_callback:
                            self.capture_callback(serial)  # 触发回调传递序列号，向主机发送切换请求

                        frame_counts[serial] += 1
                        if all(cnt >= self.max_frames for cnt in frame_counts.values()):    # 拍摄10张照片
                            break

                finally:
                    res.Release()  
            self._save_images()              

        except py.TimeoutException:
            print("采集超时，请检查相机连接")
        finally:
            self.cam_array.StopGrabbing()

    def _save_images(self):
        for serial, imgs in self.img_buffers.items():
            root_dir = os.path.join(os.getcwd(), str(serial))
            os.makedirs(root_dir, exist_ok=True)

            # 查找当前最大的 pos 索引
            existing_indices = []
            for name in os.listdir(root_dir):
                if name.startswith("pos"):
                    try:
                        index = int(name[3:])
                        existing_indices.append(index)
                    except ValueError:
                        continue
            next_idx = max(existing_indices) + 1 if existing_indices else 0

            # 创建新目录
            save_dir = os.path.join(root_dir, f"pos{next_idx}")
            os.makedirs(save_dir, exist_ok=True)

            # 保存图像
            for img_idx, img in enumerate(imgs):
                cv2.imwrite(os.path.join(save_dir, f"{img_idx:04d}.png"), img)
            print(f"相机 {serial} 已保存 {len(imgs)} 张图像")
            self.img_buffers[serial] = []


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
    
    return Camera(
        exposure_time=exposure,
        max_frames=max_frames,
        capture_callback=_capture_callback
    ), tracker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--delay', type=int, default = 1000)
    args = parser.parse_args()    
    
    while True:
        try:                 
            with SocketServer(host='10.18.18.11', base_port=4096, retries=5) as server:
                camera, tracker = create_camera_with_callback(server)
                # tracker.expected_serials = set(camera.img_buffers.keys())

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
            time.sleep(1)  # 适当增加重连间隔

