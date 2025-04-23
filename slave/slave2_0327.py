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

    
def delay_ms(ms):
    start_time = time.perf_counter()
    end_time = start_time + ms / 1000  # ms to second

    while time.perf_counter() < end_time:
        pass
 

CAMERA_NUM = 5      #多少个点位计算一次，需要从别的位置import，必须放在main外面

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--delay', type=int, default = 60)
    args = parser.parse_args()
    
    CONNECTION_FLAG = 0     #避免重复进行socket通信
    
    CAMERA_FLAG = 0     #当前采集点位的数量
    
    
    while 1:
        if CONNECTION_FLAG == 0:
            
            image_names= ["zhj","gc0","gc1","gc2","gc3","gc4","sin0","sin1","sin2","sin3"]#存储图像的命名顺序
            #image_names = [i for i in range(1, 31)]
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
            
            CONNECTION_FLAG = 1

        while(1):
            img_buffer = []       #存放一组图像的缓冲
        
            data = connection.recv(1024) #接受信息并存储到data中，设置信息最大为x个字节
            if not data:
                server_socket.close()      #加不加好像没啥影响
                connection.close()
                CONNECTION_FLAG = 0
                break

            host_order = data.decode('utf-8')
            print(f"接收到的数据: {data.decode('utf-8')}")#打印出接收到的数据并解码
            CAPTURE_CNT = 0

            #接收到的命令为"capture_order"就进行相机拍照并存储到指定位置
            if host_order == "capture_order":
                print('收到主机开始拍照命令') 
                cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                while cam.IsGrabbing():


                    # t0 = time.time() 
                    grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)     #等待一个图像，然后检索它。超时时间为5000ms。
                    # t1 = time.time() 

                    if grabResult.GrabSucceeded():                                                  # 如果图片获取成功
                        img = grabResult.Array     
                        img_buffer.append(img)
                        delay_ms(args.delay)

                        #cv2.imwrite('./output/' + str(CAPTURE_CNT+1) + '.png',img)                       
                        message = "switch_pattern"
                        connection.sendall(message.encode('utf-8'))                #发送指令0，告诉主机切换图像

                        CAPTURE_CNT += 1
                        if CAPTURE_CNT == 10:        #九张图像采集完成之后，直接跳过下面，回到等待主机capture命令语句，否则会卡死
                            pos_index = 1
                            output_folder = os.path.join(os.path.dirname(__file__),f"output/Camera/pos{pos_index}")  # 构造输出文件夹名称
                            while os.path.exists(output_folder):  # 如果输出文件夹已存在，则递增索引直到找到一个不存在的文件夹
                                pos_index += 1
                                output_folder = os.path.join(os.path.dirname(__file__),f"output/Camera/pos{pos_index}")
                            os.makedirs(output_folder)  # 创建输出文件夹                            
            
                            # 保存图像到输出文件夹中
                            for i in range(1, len(img_buffer)):
                                # 构造图像的文件名
                                image_name = image_names[i]
                                image_path = os.path.join(output_folder, f"{image_name}.png")

                                # 保存图像
                                cv2.imwrite(image_path, img_buffer[i])
                                
                            #存储完毕CAMERA_FLAG+1
                            CAMERA_FLAG+=1
                            print('CAMERA_FLAG:',CAMERA_FLAG)
                            
                            file_path = os.path.join(os.path.dirname(__file__),'FLAG.txt')           # txt文件路径         
                            with open(file_path, 'w') as file:
                                file.write(f'CAMERA_FLAG={CAMERA_FLAG}\n')              # txt写入文件

                            print(f'CAMERA_FLAG={CAMERA_FLAG} 已写入 {file_path}')    
                                
                            break

                        # t2 = time.time()

                        while 1:
                            #delay_ms(40)
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

                        #delay_ms(50)
                            
                        
                        # t3 = time.time()
                        # print(f'grabResult cost:{(t1-t0)*1000}ms')
                        # print(f'if GrabSucceeded cost:{(t2-t1)*1000}ms')
                        # print(f'while 1 cost::{(t3-t2)*1000}ms')
                    grabResult.Release()
            cam.StopGrabbing()


                

