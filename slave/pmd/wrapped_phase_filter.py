import torchvision 
import math
import time
import os
import numpy as np
import cv2 as cv
import torch
from concurrent.futures import ThreadPoolExecutor


'''
功能：计算折叠相位
按下s键自动保存为png格式。不要在python的页面保存，这样保存的图像不清晰
最后编辑时间：2023.8.25.21
'''


class WrappedPhase():
    '''用于求解包裹相位
    
    包含获取图像相位图、计算折叠相位两个方法

    Attributes：
    n:图像的数量（n步相移法）
    datapath:相机采集的相移图所在的文件夹
    width:相机采集图像的宽
    height:相机采集图像的高
    '''

    def __init__(self,datapath,width:int = 2448,height:int = 2048,n:int = 4):#12801024
        self.n = n
        self.datapath = datapath
        self.width  = width
        self.height = height
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cuda")
        #self.device = torch.device("cpu")
    
    
    def getImageData(self, m: int = 4):
        '''获取相机拍摄的n幅相移图,采用四部相移法
        return:
            I:type=ndarray,shape=(4, Height, Width),dtype=np.uint8
        '''
        I = np.empty((self.n, 2048, 2448), dtype=np.uint8)  # 预分配数组

        for i in range(m):
            filename = os.path.join(self.datapath, f"sin{i}.png")
            img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
            #img = cv.GaussianBlur(img, (3,3), 1)
            I[i] = img

        return I

    
    # def pje_filter(self,pha):
    #     '''对周期跳变进行滤波
    #     由于代码中的相对相位范围为0-2π，因此滤波计算前需要先减去π处理过零情况
        
    #     Args:
    #     pha为展开后的相对相位
        
    #     return:
    #     经过滤波的相对相位
    #     '''
    #     N = self.height # 相位图的高
    #     M = self.width # 相位图的宽
    #     n = self.n  # 判定是否跳变的步数，取1-10之间
        
    #     #############################################################################
    #     pha_temp = np.zeros((N-n, M),np.float32)
    #     pha_temp_1 = np.zeros((N-n, M),np.float32)
    #     pha_temp_n = np.zeros((N-n, M), np.float32)
    #     pha_temp = pha[0:(N-n),:]
    #     pha_temp_1[0,:] = pha[-1,:]
    #     pha_temp_1[1:, :] = pha[0:(N-n-1),:]
    #     pha_temp_n = pha[n:,:]
    #     idx = (((pha_temp_1 - math.pi) * (pha_temp - math.pi)) < 0) & (((pha_temp_n - math.pi) * (pha_temp - math.pi)) < 0)
    #     pha_temp[idx] = 2 * math.pi - pha_temp[idx]
    #     pha[0:(N-n),:] = pha_temp
    #     #############################################################################

    #     return pha   #返回经过滤波的相对相位
    
    # def computeWrappedphase(self,I):#相机采集图像的分辨率
    #     '''计算包裹相位,并绘制相位图,按s键保存
        
    #     Args:
    #     I:相机捕捉的相位图列表
    #     width,height:照片的尺寸

    #     return:真实相位[0,2*pi],pha.shape(width,height),type:ndarray
        
    #     '''
    #     #time0 = time.time()
        
    #     i0 = I[0].astype(np.float32)#转换数组I中元素的数据类型
    #     i1 = I[1].astype(np.float32)
    #     i2 = I[2].astype(np.float32)
    #     i3 = I[3].astype(np.float32)
    #     pha = np.zeros((self.height,self.width),np.float32)
    #     #print(i0.shape)
    #     #print(pha.shape)
        
    #     #############################################################################
    #     idx1 = (i0 == i2) & (i3 < i1)     #四个特殊位置
    #     idx2 = (i0 == i2) & (i3 > i1)         #四个特殊位置
    #     idx3 = (i3 == i1) & (i0 < i2)     #四个特殊位置
    #     idx4 = (i3 == i1) & (i0 > i2)     #四个特殊位置
    #     idx5 = (i0 > i2) & (i1 < i3)        # 第一象限
    #     idx6 = (i0 < i2) & (i1 < i3)        # 第二象限
    #     idx7 = (i0 < i2) & (i1 > i3)        # 第三象限
    #     idx8 = (i0 > i2) & (i1 > i3)  # 第四象限
    #     pha[idx1] = 3*math.pi/2
    #     pha[idx2] = math.pi/2
    #     pha[idx3] = math.pi
    #     pha[idx4] = 0
    #     pha[idx5] = np.arctan((i3[idx5] - i1[idx5]) / (i0[idx5] - i2[idx5]))
    #     pha[idx6] = math.pi - np.arctan((i3[idx6] - i1[idx6]) / (i2[idx6] - i0[idx6]))
    #     pha[idx7] = math.pi + np.arctan((i3[idx7] - i1[idx7]) / (i0[idx7] - i2[idx7]))
    #     pha[idx8] = 2*math.pi - np.arctan((i1[idx8] - i3[idx8]) / (i0[idx8] - i2[idx8]))
    #     # pha = np.arctan2(i1-i3, i0-i2)        使用这个会导致级次校正出错
    #     pha = self.pje_filter(pha)                       #进行相位跳变滤波
    #     #time1 = time.time()
    #     #print('时间：',time1-time0)
        
    #     pha_scaled = pha*255/(2*math.pi)            #将pha转换到为图像灰度尺度
    #     pha_scaled1 = pha_scaled.astype(np.uint8)   #转化为uint8格式，下面画图
        
    #     # if __name__ == "__main__":
    #     #     cv.imshow("Wrapped_Phase",pha_scaled1)
    #     #     key = cv.waitKey(0)
    #     #     if key == ord("s"):#按s键保存
    #     #         cv.imwrite('.\output' + r"\Wrapped_Phase_filter.png",pha_scaled1)
    #     #     cv.destroyAllWindows()
    #     # else:
    #     #     #caller = inspect.stack()[1].function
    #     #     #if caller == 'Unwrapped_phase':
            
    #     #     # import Unwrapped_phase 
    #     #     # if Unwrapped_phase.mode == 'single':
    #     #     #     cv.imwrite('.\output' + r"\Wrapped_Phase_filter1.png",pha_scaled1)
    #     #     # else:
    #     #     #     pass
            
    #     #     cv.imwrite('.\output' + r"\Wrapped_Phase_filter1.png",pha_scaled)
    #     cv.imwrite('.\output' + r"\Wrapped_Phase_filter1.png",pha_scaled)
    #     return pha


    def pje_filter(self, pha):
        N = self.height  # 相位图的高
        #M = self.width  # 相位图的宽
        n = self.n  # 判定是否跳变的步数，取1-10之间

        #pha = pha.to(self.device)
        #############################################################################
        pha_temp = pha[0:(N-n), :]
        pha_temp_1 = torch.cat([pha[-1, :].unsqueeze(0), pha[0:(N-n-1), :]])
        pha_temp_n = pha[n:, :]

        idx = ((pha_temp_1 - math.pi) * (pha_temp - math.pi) < 0) & ((pha_temp_n - math.pi) * (pha_temp - math.pi) < 0)
        pha_temp[idx] = 2 * math.pi - pha_temp[idx]
        pha[0:(N-n), :] = pha_temp
        #############################################################################

        return pha


    def computeWrappedphase(self, I):
        '''计算包裹相位,并绘制相位图,按s键保存

        Args:
        I: 相机捕捉的相位图数组
        width, height: 照片的尺寸

        return: 真实相位[0,2*pi], pha.shape(width, height), tensor,cuda:0

        '''
        
        i0 = torch.tensor(I[0], dtype=torch.float32).to(self.device)
        i1 = torch.tensor(I[1], dtype=torch.float32).to(self.device)
        i2 = torch.tensor(I[2], dtype=torch.float32).to(self.device)
        i3 = torch.tensor(I[3], dtype=torch.float32).to(self.device)
    
        

        
        pha = torch.zeros((self.height, self.width), dtype=torch.float32).to(self.device)
        
        #############################################################################
        idx1 = (i0 == i2) & (i3 < i1)  # 四个特殊位置
        idx2 = (i0 == i2) & (i3 > i1)  # 四个特殊位置
        idx3 = (i3 == i1) & (i0 < i2)  # 四个特殊位置
        idx4 = (i3 == i1) & (i0 > i2)  # 四个特殊位置
        idx5 = (i0 > i2) & (i1 < i3)  # 第一象限
        idx6 = (i0 < i2) & (i1 < i3)  # 第二象限
        idx7 = (i0 < i2) & (i1 > i3)  # 第三象限
        idx8 = (i0 > i2) & (i1 > i3)  # 第四象限
        pha[idx1] = 3 * math.pi / 2
        pha[idx2] = math.pi / 2
        pha[idx3] = math.pi
        pha[idx4] = 0
        pha[idx5] = torch.atan((i3[idx5] - i1[idx5]) / (i0[idx5] - i2[idx5]))
        pha[idx6] = math.pi - torch.atan((i3[idx6] - i1[idx6]) / (i2[idx6] - i0[idx6]))
        pha[idx7] = math.pi + torch.atan((i3[idx7] - i1[idx7]) / (i0[idx7] - i2[idx7]))
        pha[idx8] = 2 * math.pi - torch.atan((i1[idx8] - i3[idx8]) / (i0[idx8] - i2[idx8]))
    
        
        pha = self.pje_filter(pha)  # 进行相位跳变滤波,pha-.tensor,cuda:0
        
    
        return pha





if __name__ == "__main__":
    
    t0 = time.time()
    datapath = r"D:\datapath\photo\10151715"
    w = WrappedPhase(datapath)

    I = w.getImageData()
    t1 = time.time()
    print('初始化时间：',t1 - t0)    
    
    pha = w.computeWrappedphase(I) #pha是真实的折叠相位
    t2 = time.time()
    print('计算折叠相位时间：',t2 - t1)
    
    pha_scaled = pha * 255 / (2 * math.pi)  # 将pha转换到为图像灰度尺度
    pha_scaled1 = pha_scaled.cpu().numpy().astype(np.uint8)
    cv.imwrite('.\output' + r"\Wrapped_Phase_filter1.png",pha_scaled1)
    t3 = time.time()
    print('图像保存时间：',t3-t2)
    #torchvision.utils.save_image('.\output' + pha_scaled, 'Wrapped_Phase_filter.png')
    
    
    
    # for i in range(50):
    #     w.computeWrappedphase1(I)
        

    
    
