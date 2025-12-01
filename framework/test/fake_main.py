# -*-coding:utf-8 -*-
import cv2
import os
from framework.engine.main_api import PipelineExecutor

if __name__ == "__main__":
    datapath1 = r'D:\Github\_New_System\test\pos1'
    datapath2 = r'D:\Github\_New_System\test\pos2'
    imgs1 = [cv2.imread(os.path.join(datapath1, img), cv2.IMREAD_GRAYSCALE) 
                for img in sorted(os.listdir(datapath1))]
    imgs2 = [cv2.imread(os.path.join(datapath2, img), cv2.IMREAD_GRAYSCALE) 
                for img in sorted(os.listdir(datapath2))] 
    
    # 测试用例
    executor = PipelineExecutor()
    test_images = [imgs1, imgs2]     # 替换为实际图像数据
    # import pdb; pdb.set_trace()
    executor.execute_pipeline(test_images, debug=True)
