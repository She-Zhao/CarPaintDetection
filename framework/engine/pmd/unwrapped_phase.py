#import inspect
from pathlib import Path
import os
import time
import queue
import numpy as np
import cv2 as cv
import threading
import torch
import torchvision.transforms.functional as TF
from gc_binarization import Binariization
from wrapped_phase import WrappedPhase 
from threading import Thread

#global mode
#mode = 'single'


class Unwrappedphase():
    '''解包裹相位
    
    Attributes:
        n:图片数量
        row,col:相机采集图像的行和列（行小）
    '''
    def __init__(self):
        # self.datapath=datapath
        # self.row=row
        # self.col=col
        self.n = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cuda")
        #self.device = torch.device("cpu")
 
    # def gray_to_series(self, BCG):
    #     '''将格雷码转换为级次图像，并保存级次图像

    #     Args:
    #         BCG: 相机格雷码黑白(0 255)图像, tensor,cuda:0, .shape(4,row,col)

    #     Returns:
    #         series，series: 周期级次图像,tensor,cuda:0
    #     '''
    #     #BCG = torch.from_numpy((BCG / 255).astype(np.uint8))
    #     BCG = (BCG / 255).to(torch.uint8)
    #     # print(type(BCG))
    #     # print(BCG.dtype)
    #     # print(BCG.device)
    #     rows, cols = BCG[0].shape
    #     series = torch.zeros((rows, cols), dtype=torch.uint8).to(self.device)
    #     series1 = torch.zeros((rows, cols), dtype=torch.uint8).to(self.device)
    #     bin_img = torch.zeros_like(BCG).to(self.device)
    #     bin_img[0] = BCG[0]

    #     for i in range(1, self.n):
    #         bin_img[i] = torch.bitwise_xor(BCG[i], bin_img[i - 1])

    #     for j in range(0, self.n):
    #         if self.n - j - 2 >= 0:
    #             series += 2**(self.n - j - 2) * bin_img[j]
    #         if self.n - j - 1 >= 0:
    #             series1 += 2**(self.n - j - 1) * bin_img[j]

    #     series1 = torch.floor((series1 + 1) / 2)

    #     series_scale = (series * (255 / 2**(self.n - 1) - 1)).to(torch.uint8)
    #     cv.imwrite('.\output\series.png', series_scale.cpu().numpy())

    #     return series, series1    

    def get_absphase(self, wrappedphase, series, series1):
        '''求取绝对相位
        return:     unwrapped_pha -> tensor cuda:0
        '''

        unwrapped_pha = torch.zeros_like(series, dtype=torch.float32, device=self.device)

        idx1 = wrappedphase <= (torch.pi / 2)
        idx2 = (wrappedphase > (torch.pi / 2)) & (wrappedphase < (3 * torch.pi / 2))
        idx3 = wrappedphase >= (3 * torch.pi / 2)
        
        unwrapped_pha[idx1] = wrappedphase[idx1] + series1[idx1] * 2 * torch.pi
        unwrapped_pha[idx2] = wrappedphase[idx2] + series[idx2] * 2 * torch.pi
        unwrapped_pha[idx3] = wrappedphase[idx3] + (series1[idx3] - 1) * 2 * torch.pi

        return unwrapped_pha
    
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
        series,series1 = U.gray_to_series1(GC)         #获得计算级数
        q.put(series)
        q.put(series1)

def compute_phase(datapath):
    '''计算最终的相位，折叠相位计算和格雷码并行计算'''
    begin = time.time()
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
    
    time1 = time.time()
    print("折叠相位和格雷码二值化时间：",time1-begin)
    
    '''计算绝对相位'''
    absphase = U.get_absphase(series,series1,wph)                           #相位的真实值
    absphase_scale = ((absphase*255)/(2**U.n*np.pi)).to(torch.uint8)       #映射到灰度值
    
    '''保存最终结果'''   
    cv.imwrite('.\output\Absolute_pha.png' , absphase_scale.numpy())
    time2 = time.time()
    print("绝对相位计算时间：",time2-time1)
    

if __name__ == "__main__":
    root_dir = r"/home/nvidia/Project/pmd_adjust/test_imgs"
    if torch.cuda.is_available():
        torch.zeros(1).cuda()  # 触发CUDA初始化    
        
    for idx, dir in enumerate(os.listdir(root_dir)):
        datapath = os.path.join(root_dir, dir)
        img_list = [cv.imread(os.path.join(datapath, img), cv.IMREAD_GRAYSCALE) 
                    for img in sorted(os.listdir(datapath))]

        t0 = time.time()
        W = WrappedPhase(imgs=img_list)
        wph = W.computeWrappedphase()
        t1 = time.time()
        print(f"折叠相位时间{t1-t0}s")

        B = Binariization(imgs=img_list)
        series, series1 = B.get_series()
        t2 = time.time()
        print(f"条纹级数计算时间{t2-t1}s")

        U = Unwrappedphase() 
        absphase = U.get_absphase(wph, series, series1)
        t3 = time.time()
        print(f"绝对相位计算时间{t3-t2}s")  

        time.sleep(2)
        # save_dir = os.path.join(Path(__file__).parent.name, 'output', f'{idx}')
        # os.makedirs(save_dir, exist_ok=True)

        # absphase_scale = ((absphase*255)/(2**U.n*np.pi)).to(torch.uint8)       #映射到灰度值
        # cv.imwrite(os.path.join(save_dir, f'Abs_Phase.png') , absphase_scale.cpu().numpy())
        
        # # 保存折叠相位
        # pha_scaled = wph * 255 / (2 * torch.pi)  # 将pha转换到为图像灰度尺度
        # pha_scaled1 = pha_scaled.cpu().numpy().astype(np.uint8)
        # cv.imwrite(os.path.join(save_dir, f'Wrapped_Phase.png'), pha_scaled1)

        # # 保存条纹级数
        # series_scale = (series * (255 / 2**(B.n - 1) - 1)).to(torch.uint8)
        # cv.imwrite(os.path.join(save_dir, f'series.png'), series_scale.cpu().numpy())     



    # datapath = r"/home/nvidia/Project/pmd_adjust/test_imgs/pos1"
    # t0 = time.time()
    # W = WrappedPhase(datapath)
    # wph = W.computeWrappedphase()
    # t1 = time.time()
    # print(f"折叠相位时间{t1-t0}s")

    # B = Binariization(datapath)
    # series, series1 = B.get_series1()
    # t2 = time.time()
    # print(f"条纹级数计算时间{t2-t1}s")

    # U = Unwrappedphase() 
    # absphase = U.get_absphase(wph, series, series1)
    # t3 = time.time()
    # print(f"绝对相位计算时间{t3-t2}s")  

    # absphase_scale = ((absphase*255)/(2**U.n*np.pi)).to(torch.uint8)       #映射到灰度值
    # save_path_abs = os.path.join(Path(__file__).parent.name, 'output', f'Abs_Phase.png')
    # print(save_path_abs)
    # cv.imwrite(save_path_abs , absphase_scale.cpu().numpy())
    
    # # 保存折叠相位
    # pha_scaled = wph * 255 / (2 * torch.pi)  # 将pha转换到为图像灰度尺度
    # pha_scaled1 = pha_scaled.cpu().numpy().astype(np.uint8)
    # save_path_wph = os.path.join(Path(__file__).parent.name, 'output', f'Wrapped_Phase.png')
    # cv.imwrite(save_path_wph, pha_scaled1)

    # # 保存条纹级数
    # series_scale = (series * (255 / 2**(B.n - 1) - 1)).to(torch.uint8)
    # save_path_series = os.path.join(Path(__file__).parent.name, 'output', f'series.png')
    # cv.imwrite(save_path_series, series_scale.cpu().numpy())

    






# if __name__ == "__main__":
    # t0 =time.time()

    # datapath = r"D:\datapath\photo\10151715"
    # W = WrappedPhase(datapath)
    # B = Binariization(datapath)
    # U = Unwrappedphase(datapath) 
    
    # t1 =time.time()
    # print('初始化时间：',t1-t0)
    
    # #计算折叠相位
    # wph = W.computeWrappedphase()
    # pha_scaled = wph * 255 / (2 * torch.pi)  # 将pha转换到为图像灰度尺度
    # pha_scaled1 = pha_scaled.cpu().numpy().astype(np.uint8)
    # cv.imwrite('.\output' + r"\Wrapped_Phase_filter1.png",pha_scaled1)
    
    # t2 =time.time()
    # print('折叠相位计算时间：',t2-t1)
    
    # #格雷码二值化
    # #gc = B.get_Binary_gc(-2)
    # gc = B.get_Binary_wph(12)
    # gc1=gc.cpu().numpy()
    # for u in range(B.n):
    #     cv.imwrite('.\output' + '\Binarized_GC-' + str(u) + ".png",gc1[u])
        
    # t3 =time.time()
    # print('二值化时间：',t3-t2)
        
    # #计算绝对相位
    # series,series1 = U.gray_to_series(gc)
    # absphase = U.get_absphase(series,series1,wph)
    # absphase_scale = ((absphase*255)/(2**U.n*np.pi)).to(torch.uint8)       #映射到灰度值
    # cv.imwrite('.\output\Absolute_pha.png' , absphase_scale.cpu().numpy())
    # cv.imshow('mat', absphase_scale.cpu().numpy())
    # k = cv.waitKey()
    
    # t4 =time.time()
    # print('绝对相位计算时间：',t4-t3)
    # print('全过程时间：',t4-t0)
