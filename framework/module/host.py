"""
文件: Host.py
功能: 主机控制系统，负责管理投影仪和从机通信
依赖: 
- screeninfo (获取显示器信息)
- numpy (图像数据处理)
- opencv-python (图像显示和读取)
- socket (网络通信)
- atexit (程序退出时自动清理)

典型用法:
>>> host = HostControl('./projection_images/')
>>> host.Socket_init()  # 初始化从机连接
>>> host.Take_photo()   # 执行拍照流程
"""

import screeninfo
import numpy as np
import os
import socket
import cv2
import time
import atexit

class HostControl():
    """主机控制核心类，负责协调投影仪和从机相机
    
    主要功能层级:
    ├─ 初始化系统:
    │    ├─ 加载投影图像 (Projected_init)
    │    └─ 建立从机连接 (Socket_init)
    ├─ 核心操作:
    │    ├─ 控制拍照流程 (Take_photo)
    │    ├─ 安全数据发送 (_safe_send)
    │    └─ 安全数据接收 (_safe_recv)
    └─ 系统清理:
         └─ 自动资源释放 (Disconnect)

    核心特性:
    - 自动全屏投影图像序列
    - 带重试机制的从机连接
    - 严格的通信协议验证
    - 异常安全的资源管理
    """    
    def __init__(self,folder_path):
        """初始化主机控制器
        Args:
            folder_path (str): 投影图像目录路径，要求图像按数字命名(如1.png, 2.png)
        """        
        self.folder_path = folder_path
        self.server_host_1 = '10.18.18.11' #'从机IP地址四个从机可以输入四个'
        self.server_port = 4096  # 从机监听的端口号
        
        self.client_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)#设置socket工作模式
        self.image = self.Projected_init()
        self.image_nums = len(self.image)
        atexit.register(self.Disconnect)  # 新增代码

        
    def Projected_init(self):
        """初始化投影系统 (支持双屏自动识别)"""        
        # 1. 加载图像 (保持原有逻辑)
        img = []
        if not os.path.exists(self.folder_path):
             # 增加一个简单的容错，防止路径不对报错
             raise FileNotFoundError(f"投影路径不存在: {self.folder_path}")

        files = os.listdir(self.folder_path)
        files = sorted(files, key=lambda x: int(x.split('.')[0]))
        for filename in files:
            file_path = os.path.join(self.folder_path, filename)
            # 确保读取成功
            img_buffer = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img_buffer is not None:
                img.append(img_buffer)
        
        if not img:
            raise RuntimeError("未找到投影图片")
            
        image = np.array(img)
        self.image = image # 确保类属性被赋值，供 Take_photo 使用
        self.image_nums = len(self.image)
        
        # 2. 获取屏幕信息 (修改部分)
        monitors = screeninfo.get_monitors()
        if not monitors:
            raise RuntimeError("未检测到显示器")

        # 默认使用主屏 (Index 0)
        target_monitor = monitors[0]
        
        # 如果检测到多个屏幕，优先使用第二个屏幕 (Index 1) 作为投影屏
        if len(monitors) > 1:
            target_monitor = monitors[1]
            print(f"[Display] 检测到多显示器，将在扩展屏投影: {target_monitor.name}")
        else:
            print(f"[Display] 仅检测到主屏，将在主屏投影")

        # 3. 创建并移动窗口
        # 使用 WINDOW_NORMAL 允许改变窗口大小和位置
        cv2.namedWindow('projector', cv2.WINDOW_NORMAL) 
        
        # 移动窗口到目标屏幕的起始坐标 (x, y)
        cv2.moveWindow('projector', target_monitor.x, target_monitor.y)
        
        # 设置全屏
        cv2.setWindowProperty('projector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # 显示第一帧
        cv2.imshow('projector', image[0])
        cv2.waitKey(100) # 给一点时间让窗口重绘
        
        return image
    
    
    def Socket_init(self, max_retries=3, retry_interval=2):
        """初始化从机Socket连接
        Args:
            max_retries (int): 最大重试次数，默认3次
            retry_interval (int): 重试间隔(秒)，默认2秒
            
        Raises:
            RuntimeError: 超过重试次数仍未连接成功时抛出
        """        
        retries = 0
        while retries < max_retries:
            try:
                self.client_socket_1.connect((self.server_host_1, self.server_port))
                print("连接成功")
                return
            except (ConnectionRefusedError, socket.timeout) as e:
                print(f"连接失败 ({retries+1}/{max_retries}): {e}")
                retries += 1
                time.sleep(retry_interval)
            except Exception as e:
                print(f"未知错误: {e}")
                break
        raise RuntimeError("无法连接到从机")
        
    def Take_photo(self):
        """执行完整的拍照流程
        协议流程:
        1. 投影图像切换 (500ms间隔)
        2. 发送拍照指令 (capture_order)
        3. 验证从机响应 (必须为switch_pattern)
        
        Raises:
            RuntimeError: 协议验证失败时抛出
        """        
        for j in range(self.image_nums):
            cv2.imshow('projector', self.image[j])
            k = cv2.waitKey(500)
            # 发送命令
            message = 'capture_order'
            self._safe_send(message)  # 封装发送方法
            
            # 等待从机响应
            response = self._safe_recv()
            # print(f"收到从机指令 {response}")
            if response != "switch_pattern":
                raise RuntimeError(f"协议错误，期望 'switch_pattern'，收到 '{response}'")
            print(f"收到从机指令 {response}")
        self._safe_send('capture_order')

    def _safe_send(self, message):
        """(内部方法)安全发送指令
        Args:
            message (str): UTF-8编码的指令字符串
            
        Raises:
            RuntimeError: 连接异常时抛出
        """        
        try:
            self.client_socket_1.sendall(message.encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError) as e:
            raise RuntimeError("连接已中断") from e

    def _safe_recv(self):
        """(内部方法)安全接收响应
        Returns:
            str: 从机返回的UTF-8解码字符串
            
        Raises:
            RuntimeError: 接收超时或连接中断时抛出
        """        
        try:
            data = self.client_socket_1.recv(1024)
            if not data:
                raise RuntimeError("从机关闭连接")
            return data.decode('utf-8')
        except socket.timeout:
            raise RuntimeError("等待响应超时")

    def Disconnect(self):
        """安全释放所有资源
        清理顺序:
        1. 关闭Socket连接
        2. 销毁OpenCV窗口
        3. 重置连接状态标志
        
        注意: 通过atexit自动调用，无需手动执行
        """        
        try:
            # 添加socket状态判断
            if hasattr(self, 'client_socket_1') and self.client_socket_1:
                self.client_socket_1.close()
                print("Socket连接已关闭")
        except Exception as e:
            print(f"关闭socket时出错: {e}")
        finally:
            try:
                cv2.destroyAllWindows()
                print("OpenCV窗口已关闭")
            except Exception as e:
                print(f"关闭窗口时出错: {e}")