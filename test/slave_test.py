# -*-coding:utf-8 -*-
import numpy as np
import os
import socket
from pypylon import pylon
import cv2
import time
import argparse

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

if __name__ == '__main__':

    CONNECTION_FLAG = 0     #避免重复进行socket通信
    #初始化并打开相机
    cam1, cam1_ip = search_get_device(camera_ip='10.18.18.21')       
    cam1.Open()                                             

    cam2, cam2_ip = search_get_device(camera_ip='10.18.18.22')    
    cam2.Open()        
    
    while True:  
                                           
        if CONNECTION_FLAG == 0:
            # 设置监听的ip地址和端口(本机ip设置为10.18.18.4，端口orin都可以用4096为监听端口)
            host = '10.18.18.11'  # ip
            port = 4096  # 选择一个未被占用的
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((host, port))
            server_socket.listen(10)#端口开启监听
            print(f"服务器正在监听 {host}:{port}...")
            connection, address = server_socket.accept()#对连接请求进行响应连接，并获取对方ip地址
            
            CONNECTION_FLAG = 1

        while True:
        
            data = connection.recv(1024) #接受信息并存储到data中，设置信息最大为x个字节
            # if not data:
            #     server_socket.close()      #加不加好像没啥影响
            #     connection.close()
            #     CONNECTION_FLAG = 0
            #     time.sleep(0.1)
            #     break

            host_order = data.decode('utf-8')
            print(f"接收到的数据: {data.decode('utf-8')}")#打印出接收到的数据并解码
            CAPTURE_CNT = 0

            #接收到的命令为"capture_order"就进行相机拍照并存储到指定位置
            if host_order == "capture_order":
                print('收到主机开始拍照命令') 
                cam1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                cam2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                while cam1.IsGrabbing() and cam2.IsGrabbing():

                    grabResult1 = cam1.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)     #等待一个图像，然后检索它。超时时间为5000ms。
                    grabResult2 = cam2.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                    if grabResult1.GrabSucceeded() and grabResult2.GrabSucceeded():                                             # 如果图片获取成功     
                        time.sleep(0.2)
                
                        message = "switch_pattern"
                        connection.sendall(message.encode('utf-8'))                #发送指令0，告诉主机切换图像

                        CAPTURE_CNT += 1
                        if CAPTURE_CNT == 10:        #九张图像采集完成之后，直接跳过下面，回到等待主机capture命令语句，否则会卡死                    
                            print(f'CAPTURE_CNT={CAPTURE_CNT}')    
                            break

                        while True:
                            data = connection.recv(1024)
                            host_order = data.decode('utf-8')
                            print(f"接收到的数据: {data.decode('utf-8')}")#打印出接收到的数据并解码
                            if host_order == "capture_order" :
                                print('收到主机拍照命令')
                                break

                            elif not host_order:
                                break

                        if not host_order:
                            break
                            
                    grabResult1.Release()
                    grabResult2.Release()
            cam1.StopGrabbing()
            cam2.StopGrabbing()


                

