# -*-coding:utf-8 -*-
import cv2
import os
from framework.engine.main_api import PipelineExecutor
from framework.module.model_config import ModelConfigManager
import time
from pathlib import Path

def test():
    root_dir = Path(__file__).parent.parent.parent
    datapath1 = root_dir / 'test/pos1'
    datapath2 = root_dir / 'test/pos2'
    imgs1 = [cv2.imread(os.path.join(datapath1, img), cv2.IMREAD_GRAYSCALE).squeeze() 
                for img in sorted(os.listdir(datapath1))]
    imgs2 = [cv2.imread(os.path.join(datapath2, img), cv2.IMREAD_GRAYSCALE).squeeze()
                for img in sorted(os.listdir(datapath2))] 
    
    # 测试用例
    config = ModelConfigManager()
    executor = PipelineExecutor(config)
    test_images = [imgs1, imgs2]     # 替换为实际图像数据
    for _ in range(10):
        print(123)
        executor.execute_pipeline(test_images, debug=False)
        time.sleep(0.5)

if __name__ == "__main__":
    test()
