"""对 framework/output/posx/defects.json里面的检测结果进行可视化
"""
import json
import cv2
from pathlib import Path
import numpy as np

defect_map = {
    0: 'breakage',
    1: 'inclusion',
    2: 'scratch',    
    3: 'crater',
    4: 'run',
    5: 'bulge',    
    6: 'condensate',    
}

def draw_bbox_on_img(pos_idx : int = 0) -> np.ndarray:
    pos_dir = Path(f'framework/output/pos{pos_idx}')
    image_path = pos_dir / 'phase_1.png'
    json_path = pos_dir / f"defects.json"
    img = cv2.imread(str(image_path))        # ch=3
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for bbox in data[0]:    # 0代表第一个相机
        c, x, y, w, h, conf = bbox
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)        
        
        color = (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=1)
        text = f"{defect_map[int(c)]} : {conf:.2f}"
        
        cv2.putText(img, text, org=(x1,y1), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color, thickness=1)
    
    return img

img = draw_bbox_on_img(pos_idx=2)
cv2.imwrite('det_rlt.png', img)
# cv2.imshow('img', img)
# cv2.waitKey(0)
