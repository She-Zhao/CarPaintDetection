import torch
from framework.engine.detect.detect_api import Detectprocessor
import cv2
import numpy as np

def draw_normalize_box_on_img(img, det_rlts, input_tensor=False, stride = 32):
    """将bbox画在图像上，注意当输入模型的是tensor时，需要乘以填充后的图像大小来进行恢复bbox"""
    if img.ndim == 2:
        img = img[..., np.newaxis]
        img = np.repeat(img, 3, axis=2)
    
    draw_img = img.copy()
    img_h, img_w = draw_img.shape[:2]         # 2048, 2448
    if input_tensor:
        img_h = (img_h + stride - 1) // stride * stride
        img_w = (img_w + stride - 1) // stride * stride
    for box in det_rlts[0]:
        c, x, y, w, h, conf = box.tolist()
        
        x_orin = x*img_w
        y_orin = y*img_h
        w_orin = w*img_w
        h_orin = h*img_h
        
        x1 = int(x_orin-w_orin/2)
        y1 = int(y_orin-h_orin/2)
        x2 = int(x_orin+w_orin/2)
        y2 = int(y_orin+h_orin/2)
        
        color = (0, 0, 255)
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color=color, thickness=1)
        text = f"class id:{int(box[0])}:{conf:.2f}"
        cv2.putText(draw_img, text, (x1, y1), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=color, thickness=1)
        
    return draw_img

def draw_box_on_img(img, det_rlts, input_tensor=False, stride = 32):
    """将bbox画在图像上，注意当输入模型的是tensor时，需要乘以填充后的图像大小来进行恢复bbox"""
    if img.ndim == 2:
        img = img[..., np.newaxis]
        img = np.repeat(img, 3, axis=2)
    
    draw_img = img.copy()
    img_h, img_w = draw_img.shape[:2]         # 2048, 2448
    if input_tensor:
        img_h = (img_h + stride - 1) // stride * stride
        img_w = (img_w + stride - 1) // stride * stride
    for box in det_rlts[0]:
        c, x, y, w, h, conf = box.tolist()
        
        x1 = int(x-w/2)
        y1 = int(y-h/2)
        x2 = int(x+w/2)
        y2 = int(y+h/2)
        
        color = (0, 0, 255)
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color=color, thickness=1)
        text = f"class id:{int(box[0])}:{conf:.2f}"
        cv2.putText(draw_img, text, (x1, y1), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=color, thickness=1)
        
    return draw_img


def test_img_np():
    """直接读取ndarry进行测试"""
    img_path = r'D:\Github\_New_System\CarPaintDetection\framework\output\pos1\phase_1.png'
    img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).squeeze(-1)     # 模拟相机采集的单通道图像
    # img_tensor = torch.as_tensor(img_np, device='cuda')

    detect = Detectprocessor()
    det_rlts = detect(img_np)
    draw_img = draw_box_on_img(img_np, det_rlts)
    cv2.imwrite('det_rlts_np.png', draw_img)
    print(det_rlts)

def test_img_tensor():
    """直接读取ndarry进行测试"""
    img_path = r'D:\Github\_New_System\CarPaintDetection\framework\output\pos1\phase_1.png'
    img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).squeeze(-1)     # 模拟相机采集的单通道图像
    img_tensor = torch.as_tensor(img_np, device='cuda')

    detect = Detectprocessor()
    det_rlts = detect(img_tensor)
    draw_img = draw_box_on_img(img_np, det_rlts, input_tensor=True)
    cv2.imwrite('det_rlts_tensor.png', draw_img)
    print(det_rlts)

if __name__ == "__main__":
    test_img_np()
    test_img_tensor()
