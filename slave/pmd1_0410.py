import numpy as np
import os
import sys
import socket
from pypylon import pylon
import cv2
import time
import paramiko # 用于调用scp命令
from scp import SCPClient
import torch
import argparse
from concurrent.futures import ThreadPoolExecutor

# # PMD
sys.path.append(os.path.join(os.path.dirname(__file__),'pmd'))
from GC_binarization import Binariization
from wrapped_phase_filter import WrappedPhase 
from Unwrapped_phase import Unwrappedphase

# PMD end
th_list = []
#点位个数
point_num = 15
#读取1.txt中的阈值参数
def read_th():
    list = []
    for i in range(point_num):
        list.append([])
    with open(r'./1.txt', 'r') as file:
        file_name = []
        file_para = []
        for line in file:
            line = line.split(' ', 2)
            file_name.append(line[1])
            file_para.append(line[2])
    m = 0
    for ff in file_name:
        if ff[0] == '1':
            cal_para = eval(file_para[m])
            nn = int(ff.split('_',1)[1]) - 1
            list[nn] =cal_para
        m = m + 1
    print(list)
    return list
# 定义一个排序函数，提取每个字符串中的数字部分进行排序
def pos_sort(position):
    return int(position[3:])  # 提取字符串中的数字部分，并转换为整数


def compute_phase(datapath):
    pos_num = int(datapath.split("pos", 1)[1])
    th_flag = pos_num % point_num
    if th_flag == 0:
        th_flag = point_num-1
    else:
        th_flag = th_flag - 1

    th1 = th_list[th_flag][0]
    th2 = th_list[th_flag][1]
    th3 = th_list[th_flag][2]
    th4 = th_list[th_flag][3]
    th5 = th_list[th_flag][4]

    print(th_list[th_flag])

    W = WrappedPhase(datapath)
    B = Binariization(datapath,th1,th2,th3,th4,th5)
    U = Unwrappedphase(datapath) 

    #计算折叠相位
    I = W.getImageData()
    wph = W.computeWrappedphase(I)

    #格雷码二值化
    gc = B.get_Binary_wph(10)
        
    #计算绝对相位
    series,series1 = U.gray_to_series(gc)
    absphase = U.get_absphase(series,series1,wph)
    absphase_scale = ((absphase*255)/(2**U.n*np.pi)).to(torch.uint8)           #映射到灰度值

    #保存最终结果
    phasepath = os.path.join(os.path.dirname(__file__),"output/Absolute_pha")
    if not os.path.exists(phasepath):                       #创建保存目录
        os.makedirs(phasepath) 

    # filename = os.path.join(phasepath , 'Absolute_pha_Camera2' + datapath[16:] + '_0.png')
    filename = os.path.join(phasepath, 'Absolute_pha_Camera1_pos' + str(pos_num) + '_0.png')
    print('datapath:',datapath)
    print('filename:',filename)
    cv2.imwrite(filename, absphase_scale.cpu().numpy()) 
    return filename



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--NUM', type=int, default = 5)
    args = parser.parse_args()
    CAMERA_NUM = args.NUM
    print(f"CAMERA_NUM:{CAMERA_NUM}")
    
    # 文件路径
    file_path = os.path.join(os.path.dirname(__file__),"FLAG.txt")
    print('pmd has been started')

    th_list = read_th()
    

    while 1:
        time.sleep(0.1)
        # 读取文件
        with open(file_path, 'r') as file:
            # 读取第一行，并去除末尾的换行符
            content = file.readline().strip()

        # 解析内容，获取 CAMERA_FLAG 的值
        # 假设内容的格式为 "CAMERA_FLAG=x"
        key, value = content.split('=')
        CAMERA_FLAG = int(value)
       


        if CAMERA_FLAG:
            if CAMERA_FLAG%CAMERA_NUM == 0:
    
                #下面这部分是调用compute_phase，利用tensor在gpu上并行执行图像
                begin = time.time()
                camerapath = os.path.join(os.path.dirname(__file__),'output/Camera/')
                pos_files = os.listdir(camerapath)
                sorted_files = sorted(pos_files,key=pos_sort)
                print('sorted:',sorted_files)
                group_files = sorted_files[-CAMERA_NUM:]
                print('group:',group_files)

                with ThreadPoolExecutor(max_workers=CAMERA_NUM) as executor: #max_workers指定每次最多处理几组照片
                    # 利用线程池并行执行compute_phase1函数
                    executor.map(compute_phase, [os.path.join(camerapath, file) for file in group_files])

                end = time.time()
                t=end - begin   
                print("总时间：",t)
                print('处理帧率:',(9*CAMERA_NUM)/t)



                with open(file_path, 'w') as file:
                    file.write('CAMERA_FLAG=0\n')


'''
#下面这部分依次执行每组图像，需要保留，因为多线程工作报错不会定位哪里出问题，所以debug时要用下面这部分
    # begin = time.time()
    
    # for file in group_files:
    #     time0 = time.time()   
    #     datapath = os.path.join(camerapath, file)
    #     compute_phase(datapath)
    #     time1 = time.time() 
    #     print('单组图像计算时间：',time1-time0)

    # end = time.time()
    # print("总时间：",end - begin)
 
'''




