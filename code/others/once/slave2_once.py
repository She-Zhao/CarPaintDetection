# -*-coding:utf-8 -*-
import numpy as np
import os
import socket
from pypylon import pylon
import cv2
import time
import argparse

def search_get_device():#查找相机
    tl_factory = pylon.TlFactory.GetInstance()
    for dev_info in tl_factory.EnumerateDevices():
        if dev_info.GetIpAddress()[-1] == '2':
            print("DeviceClass:", dev_info.GetDeviceClass())
            print(f"ModelName:{dev_info.GetModelName()}\n"f"IP:{dev_info.GetIpAddress()}")
            camera = pylon.InstantCamera(tl_factory.CreateDevice(dev_info))
            break
    else:
        raise EnvironmentError("no GigE device found")
    return camera


CAMERA_NUM = 5      #多少个点位计算一次，需要从别的位置import，必须放在main外面

if __name__ == '__main__':
    save_path = '/home/nvidia/demo/slave/output/0409/cam'

    while 1:

        #初始化并打开相机
        cam = search_get_device()       #查找相机
        cam.Open()                      #将相机打开

        # 设置监听的ip地址和端口(本机ip设置为10.18.18.4，端口orin都可以用4096为监听端口)
        host = '10.18.18.12'  # ip
        port = 4096  # 选择一个未被占用的
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(10)#端口开启监听
        print(f"服务器正在监听 {host}:{port}...")
        connection, address = server_socket.accept()#对连接请求进行响应连接，并获取对方ip地址

        while(1):
        
            data = connection.recv(1024) #接受信息并存储到data中，设置信息最大为x个字节
            if not data:
                server_socket.close()      #加不加好像没啥影响
                connection.close()
                break

            host_order = data.decode('utf-8')
            print(f"接收到的数据: {data.decode('utf-8')}")#打印出接收到的数据并解码

            #接收到的命令为"capture_order"就进行相机拍照并存储到指定位置
            if host_order == "capture_order":
                print('收到主机开始拍照命令') 
                cam.StartGrabbingMax(1)
                while cam.IsGrabbing():

                    grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)     #等待一个图像，然后检索它。超时时间为5000ms。

                    if grabResult.GrabSucceeded():                                                  # 如果图片获取成功
                        img = grabResult.Array 
                        file = sorted(os.listdir(save_path))[-1]
                        aa = int(file.split("pos")[1][:-4])
                        aa = aa+1    
                        cv2.imwrite(save_path+r'/Camera2_pos'+str(aa)+'.png',img)       #######
                                               
                    grabResult.Release()
            cam.StopGrabbing()


                

