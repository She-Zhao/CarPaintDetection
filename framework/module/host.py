import screeninfo
import numpy as np
import os
import socket
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
import atexit

class Host():
    def __init__(self,folder_path):
        self.folder_path = folder_path
        self.server_host_1 = '10.18.18.11' #'从机IP地址四个从机可以输入四个'
        self.server_port = 4096  # 从机监听的端口号
        
        self.client_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)#设置socket工作模式
        self.image = self.Projected_init()
        self.image_nums = len(self.image)
        atexit.register(self.Disconnect)  # 新增代码

        
    def Projected_init(self):
        img = []
        files = os.listdir(self.folder_path)
        files = sorted(files, key=lambda x: int(x.split('.')[0]))
        for filename in files:
            file_path = os.path.join(self.folder_path, filename)             # 构建完整的文件路径
            img_buffer = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)     # 使用OpenCV读取图像
            img.append(img_buffer)                                      # 将图像添加到数组中
        image = np.array(img)
        
        # 初始化屏幕
        screen_id = 0
        is_color = False
    
        # get the size of the screen
        screen = screeninfo.get_monitors()[screen_id]
        width, height = screen.width, screen.height
        
        cv2.namedWindow('projector', cv2.WND_PROP_FULLSCREEN)                                                           
        cv2.moveWindow('projector', screen.x - 1, screen.y - 1)
        cv2.setWindowProperty('projector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('projector', image[0])
        k = cv2.waitKey(60)
        # 初始化屏幕完毕
        return image
    
    
    def Socket_init(self, max_retries=3, retry_interval=2):
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
        try:
            self.client_socket_1.sendall(message.encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError) as e:
            raise RuntimeError("连接已中断") from e

    def _safe_recv(self):
        try:
            data = self.client_socket_1.recv(1024)
            if not data:
                raise RuntimeError("从机关闭连接")
            return data.decode('utf-8')
        except socket.timeout:
            raise RuntimeError("等待响应超时")

    def Disconnect(self):
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