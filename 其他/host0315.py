# -*-coding:utf-8 -*-
import screeninfo
import numpy as np
import os
import socket
import cv2
import time
from concurrent.futures import ThreadPoolExecutor


def delay_ms(ms):
    start_time = time.perf_counter()
    end_time = start_time + ms / 1000  # ms to second

    while time.perf_counter() < end_time:
        pass
    
def Projected_img_buffer(folder_path):
    img = []
    files = os.listdir(folder_path)
    files = sorted(files)
    for filename in files:
        file_path = os.path.join(folder_path, filename)             # 构建完整的文件路径
        img_buffer = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)     # 使用OpenCV读取图像
        img.append(img_buffer)                                      # 将图像添加到数组中
    image = np.array(img)
    return image

if __name__ == '__main__':

    folder_path = os.path.join(os.path.dirname(__file__),'num')
    # folder_path = r'/home/gdzy/switch_pattern/demo0320/pattern'
    image = Projected_img_buffer(folder_path)

    # 初始化屏幕
    screen_id = 0
    is_color = False
 
    # get the size of the screen
    screen = screeninfo.get_monitors()[screen_id]
    width, height = screen.width, screen.height
    # 初始化屏幕完毕
    
    # 初始socket通信的端口连接
    server_host_1 = '10.18.18.11' #'从机IP地址四个从机可以输入四个'
    server_host_2 = '10.18.18.12' 
    server_host_3 = '10.18.18.13' 
    server_host_4 = '10.18.18.14' 
    server_port = 4096  # 从机监听的端口号
    
    client_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)#设置socket工作模式
    client_socket_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)#设置socket工作模式
    client_socket_3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)#设置socket工作模式
    client_socket_4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)#设置socket工作模式
    
    client_socket_1.connect((server_host_1, server_port))#与从机端口进行连接
    client_socket_2.connect((server_host_2, server_port))
    client_socket_3.connect((server_host_3, server_port))
    client_socket_4.connect((server_host_4, server_port))
    cnt = 1
    fps_total = 0
    cv2.namedWindow('projector', cv2.WND_PROP_FULLSCREEN)                                                           
    cv2.moveWindow('projector', screen.x - 1, screen.y - 1)
    cv2.setWindowProperty('projector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('projector', image[0])
    k = cv2.waitKey(50)

    while(1):
        
        delay_ms(3000)
        ta = time.time()
        for j in range(10):     #实际只需要9张图像
                            
            cv2.imshow('projector', image[j])
            k = cv2.waitKey(1)
            # delay_ms(31)
            
            #客户端Orin0发送给orin1-4命令
            message = 'capture_order'
            client_socket_1.sendall(message.encode('utf-8'))#发送指令1 进行拍照
            client_socket_2.sendall(message.encode('utf-8'))#发送指令1 进行拍照
            client_socket_3.sendall(message.encode('utf-8'))#发送指令1 进行拍照
            client_socket_4.sendall(message.encode('utf-8'))#发送指令1 进行拍照
            #print(f'客户端发出命令:{message}')
            
            while 1:                                                #给其他几个orin发送完命令后，循环监听，等待从orin的回复指令
                data1 = client_socket_1.recv(1024)                  #接受信息并存储到data中，设置信息最大为2个字节
                data2 = client_socket_2.recv(1024)
                data3 = client_socket_3.recv(1024)
                data4 = client_socket_4.recv(1024)
                
                slave1_respond = data1.decode('utf-8')
                slave2_respond = data2.decode('utf-8')
                slave3_respond = data3.decode('utf-8')
                slave4_respond = data4.decode('utf-8')
                print(f'客户端收到服务端响应:{slave1_respond}')
                
                if slave1_respond == "switch_pattern" and slave2_respond == "switch_pattern" and slave3_respond == "switch_pattern" and slave4_respond == "switch_pattern":
                # if slave1_respond == "switch_pattern" :   
                    break 
                
                elif (not slave1_respond) or (not slave2_respond) or (not slave3_respond) or (not slave4_respond):
                # elif (not slave1_respond):
                    break                                           
                
        tb = time.time()
        print(f'Total time:{tb-ta}')
        fps = 10/(tb-ta)
        print(f'FPS:{fps}')
        
        cv2.imshow('projector', image[0])       #显示下一张图像
        k = cv2.waitKey(1)
        print('all right')
        cnt+=1
        if cnt == 21:
            break
        
        
#client_socket_1.close()
# client_socket_2.close()
# client_socket_3.close()
# client_socket_4.close()
    


            
            
            
            

