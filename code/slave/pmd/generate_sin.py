import cv2
import numpy as np
import math
 
class PhaseShiftingCode():
    def __init__(self,n: int = 4):
        self.n = n          #n表示需要几步相移图像
 
    def toPhasePattern(self,j:int,freq:int=64,width:int=2560,hight:int=1600):
        '''生成'''
        col = np.zeros((hight),np.uint8)       #生成一个维数为width的行向量
        for i in range(hight):
            col[i] = 127.5 + 127.5 * math.cos(2 * math.pi *( i * freq / hight + j/ self.n))
        pattern = np.tile(col,(width,1)).T
        return pattern
 
if __name__ == '__main__':                                      #只在当前模块执行，其他模块导入本模块时不执行
    n = 4
    p = PhaseShiftingCode(n)
    for k in range(n):
        pattern = p.toPhasePattern(k)
        title ='PhaseShifting-' + str(k)
        cv2.imwrite(r'.\output' + '\\' + title + '.png', pattern)
