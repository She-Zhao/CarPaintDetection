# -*-coding:utf-8 -*-
import screeninfo
import numpy as np
import os
import socket
import cv2
import time
from concurrent.futures import ThreadPoolExecutor

class Host():
    def __init__(self,folder_path):
        self.folder_path = folder_path
        self.server_host_1 = '10.18.18.11' #'从机IP地址四个从机可以输入四个'
        self.server_port = 4096  # 从机监听的端口号
        
        self.client_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)#设置socket工作模式
        self.image = self.Projected_init()
        self.image_nums = len(self.image)

        
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
    
    
    def Socket_init(self):
        
        self.client_socket_1.connect((self.server_host_1, self.server_port))#与从机端口进行连接
        
    def Take_photo(self):
        for j in range(self.image_nums):     #实际只需要9张图像   
            cv2.imshow('projector', self.image[j])
            k = cv2.waitKey(1)
            
            #客户端Orin0发送给orin1-4命令
            message = 'capture_order'
            self.client_socket_1.sendall(message.encode('utf-8'))#发送指令1 进行拍照
            print(f'客户端发出命令:{message}')
            
            while 1:                                                #给其他几个orin发送完命令后，循环监听，等待从orin的回复指令
                data1 = self.client_socket_1.recv(1024)                  #接受信息并存储到data中，设置信息最大为2个字节
                slave1_respond = data1.decode('utf-8')
                print(f'客户端收到服务端响应:{slave1_respond}')
                if slave1_respond == "switch_pattern" :    
                    break 
                
                elif (not slave1_respond):
                    break
        
        cv2.imshow('projector', self.image[0])       #显示下一张图像
        k = cv2.waitKey(1)
        
    def Disconnect(self):
        self.client_socket_1.close()
        
    
if __name__ == '__main__':

    folder_path = r'D:\Project\CarPaintDetection\code\host\patterns\nums10'
    host = Host(folder_path)
    
    host.Socket_init()
    host.Projected_init()

    time.sleep(1)
    host.Take_photo()
            
    host.Disconnect()