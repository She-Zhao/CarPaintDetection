import cv2
import numpy as np

# 读取输入图像
input_image = cv2.imread(r'D:\Project\PMD\photo\Camera_0\1\gc1.', cv2.IMREAD_GRAYSCALE)

# 定义结构元素的大小，这取决于你希望检测的亮度变化的尺度
kernel_size = (3000, 3000)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

# 应用底帽变换
tophat_transformed = cv2.morphologyEx(input_image, cv2.MORPH_TOPHAT, kernel)

# 显示原始图像和底帽变换结果
cv2.imshow('Original Image', input_image)
cv2.imshow('Top-Hat Transformed', tophat_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()
