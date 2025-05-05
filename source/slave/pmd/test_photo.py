import time
import torch
import cv2
import numpy as np
import math
import os
import cv2 as cv
from GC_binarization import Binariization
from wrapped_phase_filter import WrappedPhase
from Unwrapped_phase import Unwrappedphase
import matplotlib.pyplot as plt

'''
功能：
遍历求二值化阈值的解

'''
def cal_gradient(I):
    # 计算 Sobel 梯度
    sobelx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向梯度
    sobely = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向梯度

    # 计算梯度幅值
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 计算平均梯度
    average_gradient = np.mean(gradient_magnitude)
    return average_gradient

    print("指定区域的平均梯度：", average_gradient)
def getImageData(datapath):
    I = np.empty((4, 2048, 2448), dtype=np.uint8)  # 预分配数组
    for i in range(4):
        filename = os.path.join(datapath, f"sin{i}.png")
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        I[i] = img
    return I
def np8_cal(i0,i1):
    i01 = np.empty(i0.shape, dtype=np.uint8)
    idx_compare0 = i0 < i1
    idx_compare1 = i0 >= i1
    i01[idx_compare0] = i1[idx_compare0] - i0[idx_compare0]
    i01[idx_compare1] = i0[idx_compare1] - i1[idx_compare1]

    return i01


def compute_phase_cuda(datapath,th1,th2,th3,th4,th5):
    W = WrappedPhase(datapath)
    B = Binariization(datapath,th1,th2,th3,th4,th5)
    U = Unwrappedphase(datapath)

    # 计算折叠相位
    I = W.getImageData()
    wph = W.computeWrappedphase(I)

    # 格雷码二值化
    gc = B.get_Binary_wph(10)

    # 计算绝对相位
    series, series1 = U.gray_to_series(gc)
    absphase = U.get_absphase(series, series1, wph)
    absphase_scale = ((absphase * 255) / (2 ** U.n * np.pi)).to(torch.uint8)  # 映射到灰度值
    return  absphase_scale.cpu().numpy()

    # # 保存最终结果
    # savepath = datapath + r'\output'
    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)
    # cv.imwrite(savepath + '\Absolute_pha.png', absphase_scale.cpu().numpy())

    '''以下这部分是保存折叠相位、格雷码的代码，不建议加上，对速度影响较大，且有舍入误差（是显示的问题，不影响绝对相位）'''
    # pha_scaled = wph * 255 / (2 * np.pi)  # 将pha转换到为图像灰度尺度
    # pha_scaled1 = pha_scaled.cpu().numpy().astype(np.uint8)
    # cv.imwrite(savepath + r"\Wrapped_Phase_filter1.png",pha_scaled1)

    # gc1=gc.cpu().numpy()
    # for u in range(B.n):
    #     cv.imwrite(savepath + '\Binarized_GC-' + str(u) + ".png",gc1[u])


# # 自适应动态阈值分割
# def adaptiveThresh(I, winSize, ratio=0.15):
#     # 第一步:对图像矩阵进行均值平滑
#     I_mean = cv2.boxFilter(I, cv2.CV_32FC1, winSize)
#
#     # 第二步:原图像矩阵与平滑结果做差
#     out = I - (1.0 - ratio) * I_mean
#
#     # 第三步:当差值大于或等于0时，输出值为255；反之，输出值为0
#     out[out >= 0] = 255
#     out[out < 0] = 0
#     out = out.astype(np.uint8)
#     return out


# 用正弦图像像素深度变化判断是否为有效区域
def area_cal(photo_path):
    I = getImageData(photo_path)

    i01 = np8_cal(I[0], I[1])
    i12 = np8_cal(I[1], I[2])
    i23 = np8_cal(I[2], I[3])
    i30 = np8_cal(I[3], I[0])

    idx_compare = i01 < i12
    i01[idx_compare] = i12[idx_compare]
    idx_compare = i01 < i23
    i01[idx_compare] = i23[idx_compare]
    idx_compare = i01 < i30
    i01[idx_compare] = i30[idx_compare]

    # cv.imshow('mat', i01)
    # k = cv2.waitKey()
    # 查看直方图，可以看到两波峰之间的位置大约为30-50，如果采用全局阈值可以选用30-50
    # hist, bins = np.histogram(i01.flatten(), bins=256)
    # plt.bar(range(len(hist)), hist)
    # plt.show()

#全局阈值，采用分位数确定阈值，腐蚀膨胀再腐蚀
    threshold_value_0 = np.percentile(i01, 40)
    threshold_value_255 = np.percentile(i01, 40)

    idx_compare = i01 < threshold_value_0
    i01[idx_compare] = 0
    idx_compare = i01 >= threshold_value_255
    i01[idx_compare] = 255

    cv.imshow('mat', i01)
    k = cv2.waitKey()
    kernel = np.ones((6, 6), np.uint8)

    i01 = cv2.erode(i01, kernel, iterations=1)
    # cv.imshow('mat', i01)
    # k = cv2.waitKey()

    kernel = np.ones((6, 6), np.uint8)
    i01 = cv2.dilate(i01, kernel, iterations = 10)
    # cv.imshow('mat', i01)
    # k = cv2.waitKey()

    kernel = np.ones((10, 10), np.uint8)
    i01 = cv2.erode(i01, kernel, iterations=10)
    cv.imshow('mat', i01)
    k = cv2.waitKey()
##################################################
    # min_value = 10
    # gradient_value = []
    # min_th1 = 0
    # min_th2 = 0
    #
    # for k in range(9,51):
    #     for m in range(9,51):
    #         th1 = k/10
    #         th2 = m/10
    #         img_phase = compute_phase_cuda(photo_path, th1, th2, th3=1.5, th4=1, th5=1)
    #         idx_phase = (i01 == 0)
    #         img_phase_copy = img_phase
    #         img_phase_copy[idx_phase] = 0
    #         idx_phase = (i01 == 255)
    #         value_in_time = cal_gradient(img_phase[idx_phase])
    #         gradient_value.append(value_in_time)
    #         if min_value >= value_in_time:
    #             min_value = value_in_time
    #             min_th1 = th1
    #             min_th2 = th2
    #
    # print('all right')
    # print('最小平均梯度：', min_value)
    # print('最小th1：', min_th1)
    # print('最小th2：', min_th2)
######################################################################
    # aa = [4.0, 4.7, 2.2, 1.3, 1.3]
    aa = [1.7000000000000002, 1.6, 0.7, 0.6, 0.5]
    th1 = aa[0]
    th2 = aa[1]
    th3 = aa[2]
    th4 = aa[3]
    th5 = aa[4]
    img_phase = compute_phase_cuda(photo_path,th1,th2,th3,th4,th5)
    idx_phase = (i01 == 0)
    img_phase_copy = img_phase
    img_phase_copy[idx_phase] = 0
    cv.imshow('mat', img_phase_copy)
    k = cv2.waitKey()
    idx_phase = (i01 == 255)
    return cal_gradient(img_phase[idx_phase])
###################################################









if __name__ == "__main__":
    datapath = r'D:\datapath\20240307\valueable_photo\3_2'
    print(area_cal(datapath))





