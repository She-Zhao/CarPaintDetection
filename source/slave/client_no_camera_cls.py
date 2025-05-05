# -*-coding:utf-8 -*-
import numpy as np
import os
from pypylon import pylon
import cv2
import time

import socket
import argparse
from typing import Optional, Tuple

IMAGE_NAMES= ["zhj","gc0","gc1","gc2","gc3","gc4","sin0","sin1","sin2","sin3"]#存储图像的命名顺序
CAMERA_FLAG = 0     #当前采集点位的数量
CAMERA_NUM = 5      #多少个点位计算一次，需要从别的位置import，必须放在main外面

def search_get_device(camera_ip):#查找相机
    """
    寻找指定相机

    Args:
        camera_ip:相机的ip
    
    returns:
        对应相机
    """
    tl_factory = pylon.TlFactory.GetInstance()
    for dev_info in tl_factory.EnumerateDevices():
        if dev_info.GetIpAddress() == str(camera_ip):
            print("DeviceClass:", dev_info.GetDeviceClass())
            print(f"ModelName:{dev_info.GetModelName()}\n"f"IP:{dev_info.GetIpAddress()}")
            camera = pylon.InstantCamera(tl_factory.CreateDevice(dev_info))
            break
    else:
        raise EnvironmentError("no GigE device found")
    return camera, camera_ip


def save_imgs(cam_ip, img_buffer):
    pos_index = 1
    output_folder = os.path.join(os.path.dirname(__file__),f"output/Camera{cam_ip}/pos{pos_index}")  # 构造输出文件夹名称
    while os.path.exists(output_folder):
        pos_index += 1
        output_folder = os.path.join(os.path.dirname(__file__),f"output/Camera{cam_ip}/pos{pos_index}")
    os.makedirs(output_folder)  # 创建输出文件夹                           

    for i in range(1, len(img_buffer)):
        # 构造图像的文件名
        image_name = IMAGE_NAMES[i]
        image_path = os.path.join(output_folder, f"{image_name}.png")
        # 保存图像
        cv2.imwrite(image_path, img_buffer[i])
    
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

def delay_ms(ms):
    start_time = time.perf_counter()
    end_time = start_time + ms / 1000  # ms to second

    while time.perf_counter() < end_time:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--delay', type=int, default = 1000)
    args = parser.parse_args()
    
    #初始化并打开相机
    cam1, cam1_ip = search_get_device(camera_ip='10.18.18.21')       
    cam1.Open()                                             

    cam2, cam2_ip = search_get_device(camera_ip='10.18.18.22')    
    cam2.Open()        
    
    while True:
        try:                 
            with SocketServer(host='10.18.18.11', base_port=4096, retries=5) as server:
                while True:
                    img_buffer1 = []       #存放一组图像的缓冲
                    img_buffer2 = []
                    host_order = server.safe_receive()  
                    print(f"接收到的数据: {host_order}")

                    CAPTURE_CNT = 0

                #接收到的命令为"capture_order"就进行相机拍照并存储到指定位置
                    if host_order == "capture_order":
                        print('收到主机开始拍照命令') 
                        cam1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                        cam2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                    
                    # while cam1.IsGrabbing():
                        while cam1.IsGrabbing() and cam2.IsGrabbing():                        
                            grabResult1 = cam1.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)     #等待一个图像，然后检索它。超时时间为5000ms。
                            grabResult2 = cam2.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                            if grabResult1.GrabSucceeded() and grabResult2.GrabSucceeded():                                             # 如果图片获取成功     
                                img_buffer1.append(grabResult1.Array)
                                img_buffer2.append(grabResult2.Array)
                                # delay_ms(args.delay)

                                server.safe_send("switch_pattern")

                                CAPTURE_CNT += 1
                                if CAPTURE_CNT == 10:        #九张图像采集完成之后，直接跳过下面，回到等待主机capture命令语句，否则会卡死
                                    save_imgs(cam1_ip, img_buffer1)
                                    save_imgs(cam2_ip, img_buffer2)
                                    write_camera_flag()       
                                    break

                                while True:
                                    next_order = server.safe_receive()
                                    print(f"接收到的数据: {next_order}")

                                    if next_order == "capture_order":
                                        print('收到主机拍照命令')
                                        break

                            grabResult1.Release()
                            grabResult2.Release()
                        cam1.StopGrabbing()
                        cam2.StopGrabbing()

        except (RuntimeError, ConnectionError, KeyboardInterrupt) as e:  
            print(f"❌ 连接异常: {str(e)}")
            print("🕒 等待重新连接...")
            time.sleep(1)  # 适当增加重连间隔
