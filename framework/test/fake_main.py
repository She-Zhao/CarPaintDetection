# 模拟core中的文件，调用各个函数，用于测试

# -*-coding:utf-8 -*-
import os
import time
import cv2
from framework.engine.main_api import PipelineExecutor

def main():
    executor = PipelineExecutor()
    datapath1 = r'D:\Project\_New_System\test\pos1'
    datapath2 = r'D:\Project\_New_System\test\pos2'
    imgs1 = [cv2.imread(os.path.join(datapath1, img), cv2.IMREAD_GRAYSCALE) 
                for img in sorted(os.listdir(datapath1))]
    imgs2 = [cv2.imread(os.path.join(datapath2, img), cv2.IMREAD_GRAYSCALE) 
                for img in sorted(os.listdir(datapath2))] 
    raw_imgs = [imgs1, imgs2]
    while True:
        try:                     
            time.sleep(1)
            while True:
                print(f"假装已经拍到了照片≡ω≡")
                time.sleep(2)
                executor.execute_pipeline(raw_imgs)
        except (RuntimeError, ConnectionError, KeyboardInterrupt) as e:  
            print(f"❌ 连接异常: {str(e)}")
            print("🕒 等待重新连接...")
            time.sleep(1)

if __name__ == '__main__':
    main()
