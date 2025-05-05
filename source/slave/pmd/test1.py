import cv2
import numpy as np
import math
import os
import cv2 as cv
import matplotlib.pyplot as plt

def cal_gradient(I):
    # 计算 Sobel 梯度
    sobelx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向梯度
    sobely = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向梯度

    # 计算梯度幅值
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 计算平均梯度
    average_gradient = np.mean(gradient_magnitude)

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



# 用正弦图像像素深度变化判断是否为有效区域
def area_div_0(I):
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

    cv.imshow('mat', i01)
    k = cv2.waitKey()
    # numpy查看直方图，可以看到两波峰之间的位置大约为30-50，如果采用全局阈值可以选用30-50
    # hist, bins = np.histogram(i01.flatten(), bins=256)
    # plt.bar(range(len(hist)), hist)
    # plt.show()
    # plt.hist(i01.ravel(), 256)
    # plt.show()
    i01 = cv2.equalizeHist(i01)
    # plt.hist(i01.ravel(), 256)
    # plt.show()

    #全局阈值，采用分位数确定阈值，腐蚀膨胀再腐蚀
    threshold_value_0 = np.percentile(i01, 20)
    threshold_value_255 = np.percentile(i01, 20)

    idx_compare = i01 < threshold_value_0
    i01[idx_compare] = 0
    idx_compare = i01 >= threshold_value_255
    i01[idx_compare] = 255

    cv.imshow('mat', i01)
    k = cv2.waitKey()
    kernel = np.ones((8, 8), np.uint8)

    i01 = cv2.erode(i01, kernel, iterations=1)
    cv.imshow('mat', i01)
    k = cv2.waitKey()

    kernel = np.ones((6, 6), np.uint8)
    i01 = cv2.dilate(i01, kernel, iterations=12)
    cv.imshow('mat', i01)
    k = cv2.waitKey()

    kernel = np.ones((12, 12), np.uint8)
    i01 = cv2.erode(i01, kernel, iterations=10)
    cv.imshow('mat', i01)
    k = cv2.waitKey()

    kernel = np.ones((2, 2), np.uint8)
    i01 = cv2.erode(i01, kernel, iterations=5)
    cv.imshow('mat', i01)
    k = cv2.waitKey()
##################################################
    img_phase = cv.imread(r'D:\datapath\pos25\2.png', cv.IMREAD_GRAYSCALE)
    idx_phase = (i01 == 0)
    img_phase_copy = img_phase
    img_phase_copy[idx_phase] = 0
    cv.imshow('mat', img_phase_copy)
    k = cv2.waitKey()
    idx_phase = (i01 == 255)
    cal_gradient(img_phase[idx_phase])




if __name__ == "__main__":
    I = getImageData(r"D:\datapath\pos23")
    area_div_0(I)





