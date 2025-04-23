import os
import time
import numpy as np
import cv2 as cv
import threading
import queue
import torch
from concurrent.futures import ThreadPoolExecutor
from GC_binarization import Binariization
from wrapped_phase_filter import WrappedPhase 
from Unwrapped_phase import Unwrappedphase
from threading import Thread

#################################################################################
#以下是用cpu依次执行的
def worker_Wph(q):
    '''获取折叠相位'''
    lock = threading.RLock()
    with lock:
        #W = WrappedPhase(datapath)
        I = W.getImageData()
        wph = W.computeWrappedphase(I)
        q.put(wph)
    
def worker_Series(q):
    '''获取条纹级数'''
    lock = threading.RLock()
    with lock:
        #B = Binariization(datapath)
        #U = Unwrappedphase(datapath)
        GC = B.get_Binary_wph(10)
        series,series1 = U.gray_to_series(GC)         #获得计算级数
        q.put(series)
        q.put(series1)

def compute_phase_cuda_Parallel(datapath):
    '''计算最终的相位，折叠相位计算和格雷码计算分开了，未测试（开销太大，效果必定不好）
    '''
    global W,B,U
    W = WrappedPhase(datapath)
    B = Binariization(datapath)
    U = Unwrappedphase(datapath) 
    
    q = queue.Queue()       #创建一个先进先出的队列
    
    '''线程一：获取折叠相位'''
    t1 = Thread(target = worker_Wph,args=(q,))

    '''线程二： 获取条纹级数'''
    t2 = Thread(target = worker_Series,args=(q,))  
    
    # 启动线程运行
    t1.start()
    t2.start()

    # 等待所有线程执行完毕
    t1.join()  # join() 等待线程终止，要不然一直挂起
    t2.join()
    
    wph = q.get()
    series = q.get()
    series1 = q.get()
    
    '''计算绝对相位'''
    absphase = U.get_absphase(series,series1,wph)                               #相位的真实值
    absphase_scale = ((absphase*255)/(2**U.n*np.pi)).astype(np.uint8)       #映射到灰度值

    '''保存最终结果'''   
    savepath = datapath + r'\output' 
    if not os.path.exists(savepath):
        os.makedirs(savepath)  
    cv.imwrite(savepath + '\Absolute_pha.png' , absphase_scale) 
#################################################################################

  
def compute_phase_cuda(datapath):

    W = WrappedPhase(datapath)
    B = Binariization(datapath)
    U = Unwrappedphase(datapath) 
    
    #计算折叠相位
    I = W.getImageData()
    wph = W.computeWrappedphase(I)

    #格雷码二值化
    gc = B.get_Binary_wph(10)
        
    #计算绝对相位
    series,series1 = U.gray_to_series(gc)
    absphase = U.get_absphase(series,series1,wph)
    absphase_scale = ((absphase*255)/(2**U.n*np.pi)).to(torch.uint8)       #映射到灰度值
    
    #保存最终结果
    savepath = datapath + r'\output'
    if not os.path.exists(savepath):
        os.makedirs(savepath)  
    cv.imwrite(savepath + '\Absolute_pha.png' , absphase_scale.cpu().numpy()) 
    
    '''以下这部分是保存折叠相位、格雷码的代码，不建议加上，对速度影响较大，且有舍入误差（是显示的问题，不影响绝对相位）'''
    # pha_scaled = wph * 255 / (2 * np.pi)  # 将pha转换到为图像灰度尺度
    # pha_scaled1 = pha_scaled.cpu().numpy().astype(np.uint8)
    # cv.imwrite(savepath + r"\Wrapped_Phase_filter1.png",pha_scaled1)  
    
    # gc1=gc.cpu().numpy()
    # for u in range(B.n):
    #     cv.imwrite(savepath + '\Binarized_GC-' + str(u) + ".png",gc1[u])
    






if __name__ == "__main__":
    
###########################################################################  
#下面这部分是调用compute_phase_cuda，利用tensor在gpu上并行执行图像
    begin = time.time()
    
    rootpath = r"D:\Project\PMD\PMD3.0\img_200"
    files = os.listdir(rootpath)
    with ThreadPoolExecutor(max_workers=10) as executor: #max_workers指定每次最多处理几组照片
        # 利用线程池并行执行compute_phase1函数
        executor.map(compute_phase_cuda, [os.path.join(rootpath, file) for file in files])
        
    end = time.time()
    t=end - begin
    print("总时间：",t)
    print('处理帧率:',400/t)
###########################################################################  
    

###########################################################################  
#下面这部分依次执行每组图像
    # begin = time.time()
    
    # rootpath = r"D:\Project\PMD\PMD2.4\Test_IMGS"
    # files = os.listdir(rootpath)
    # for file in files:
    #     time0 = time.time()   
    #     datapath = os.path.join(rootpath, file)
    #     compute_phase_cuda(datapath)
    #     time1 = time.time() 
    #     print('单组图像计算时间：',time1-time0)

    # end = time.time()
    # print("总时间：",end - begin)
###########################################################################   



