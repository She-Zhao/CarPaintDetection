import cv2
import numpy as np
#k:把格雷码直接当成二进制对应的十进制数
#v：格雷码实际对应的十进制数
'''绘制格雷码图案
最后编辑时间:11.7
'''


class GrayCode():
    '''用于生成格雷码，并保存格雷码图案
    
    该类较为独立，函数大多为私有
    
    Attributes：
    n:生成n位格雷码
    codes:格雷码矩阵，shape = (4,16),[[0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1],[0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0],[0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0],[0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0]]
    code2k[]:16个格雷码，以"0001"这种字符串表示
    k2v[]:格雷码换成二进制码代表的值
    v2k[]:二进制码换成格雷码代表的值
    
    '''
    codes = np.array([])
    code2k = {}
    k2v = {}
    v2k = {}
    def __init__(self, n: int = 4):
        self.n = n
        self.codes = self.__formCodes(self.n)
        # 从格雷码转换到k
        for k in range(2 ** n):
            self.code2k[self.__code2k(k)] = k           #self.__code2k(k)是键，即16个格雷码；k是0~16
        # 从格雷码转换到v
        for k in range(2 ** n):
            self.k2v[k] = self.__k2v(k)                 #v表示格雷码换成二进制码代表的值
        # 从v转换到k（idx）
        for k, v in self.k2v.items():
            self.v2k[v] = k                             #v表示二进制码换成格雷码代表的值
 
    @staticmethod            #不需要实例化直接像函数一样调用（类名.方法名()来调用），定义时也不需要self，cls参数
    def __createGrayCode(n: int):
        '''生成n位格雷码'''
        if n < 1:
            print("输入数字必须大于0")
            # assert (0);
        #        elif n == 1:                                      #代码较长
        #            code = ["0", "1"]
        #            return code
        #        else:
        #            code = []
        #            code_pre = GrayCode.__createGrayCode(n - 1)   #递归嵌套
        #            code.append("0" + idx for idx in code_pre)    #解析法写for循环
        #            code.append("1" + idx for idx in code_pre(::-1))
        #            return code
        else:
            code = ["0", "1"]
            for i in range(1, n):  # 循环递归
                code_left = ["0" + idx for idx in code]  # 解析法写for循环
                code_right = ["1" + idx for idx in code[::-1]]
                code = code_left + code_right
            return code             #code[]为n位格雷码的编码数值，本例 n = ['0000', '0001', '0011', '0010', '0110', '0111', '0101', '0100', '1100', '1101', '1111', '1110', '1010', '1011', '1001', '1000']
 
    def __formCodes(self, n: int):    #两个下划线开头表示该方法只能在类内部使用
        '''生成codes矩阵'''
        code_temp = GrayCode.__createGrayCode(n)       #首先生成n位格雷码储存在code_temp中
        #print(code_temp)           #['0000', '0001', '0011', '0010', '0110', '0111', '0101', '0100', '1100', '1101', '1111', '1110', '1010', '1011', '1001', '1000']
        codes = []
        for row in range(len(code_temp[0])):           #n位格雷码循环n次，这里为4次
            c = []
            for idx in range(len(code_temp)):          #循环2**n次,这里为16次
                c.append(int(code_temp[idx][row]))     #将code_temp中第idx个元素中的第row个数添加到c中
            codes.append(c)                            #依次循环得到code_temp[idx][1]、code_temp[idx][2]、code_temp[idx][3]、code_temp[idx][4]
        return np.array(codes, np.uint8)               #得到格雷码矩阵，shape = (4,16),[[0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1],[0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0],[0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0],[0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0]]
 
    def toPattern(self, idx: int, cols: int = 1440, rows: int = 2560):
        '''生成格雷码光栅图'''
        #assert (idx >= 0) #断言用法，确保idx >= 0
        row = self.codes[idx, :]                        #self.codes[]是格雷码矩阵，idx表示codes的行索引,row为其中一副的格雷码图像
        #print(self.codes)      #[[0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1] [0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0][0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0][0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0]]
        #print(row)     #row:[0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1]
        one_row = np.zeros((cols), np.uint8)                    #生成1440行向量，one_row
        #assert (cols % len(row) == 0)
        per_col = int(cols / len(row))                          #将1280个像素分成2**n = 16份
        for i in range(len(row)):
            one_row[i * per_col: (i + 1) * per_col] = row[i]        #one_row[]为每一列的黑白，通过row赋值，此时one_row是行向量
        pattern = (np.tile(one_row, (rows, 1)) * 255).T             #np.tile(a，(b,c))函数用a来重构b行c列，这里为2560行，1440列，然后取转置即可
        return pattern                                              #2560行，1440列的格雷码图案
 
 
    def __code2k(self, k):
        '''将k映射到对应的格雷码'''
        col = self.codes[:, k]      #col为某一列格雷码图像的值，例如‘0001’
        code = ""
        for i in col:
            code += str(i)
        return code                 #col是一维数组[0,0,0,1]，code是字符串"0001"
 
    def __k2v(self, k):
        '''将k映射为v'''
        col = list(self.codes[:, k])           #将第k列格雷码储存在col中
        col = [str(i) for i in col]
        code = "".join(col)                    #将列表中的元素整合到一起
        v = int(code,2)
        return v                    #v表示格雷码换成二进制码代表的值
 
 
    def store_gray_code_map_value(self):
        '''将8幅256列的格雷码编码图的码值列表储存在'格雷光栅码值.txt'文件中'''
        filename = '格雷光栅码值.txt'
        with open(filename,'w') as fob1:
            fob1.write("codes\n")
            for i in g.codes:
                fob1.write('长度：' + str(len(i))+ '\n')
                fob1.write(str(i) + '\n')
 
    def test(self):
        '''该段代码用于测试，输入各个数值'''
        print("codes")          
        print(self.codes)
        print("\ncode -> k")
        print(self.code2k)
        print("\nk -> v")
        print(self.k2v)
        print("\nv -> k")
        print(self.v2k)
    
if __name__ == '__main__':            #只在本文件内运行一下代码
    n =  5
    g = GrayCode(n=5)
    #g.test()           #测试输出
    g.store_gray_code_map_value()
    for i in range(n):
        pattern = g.toPattern(i)            #生成四幅格雷码图案
        title ='Pattern-' + str(i)
        #cv2.imshow(title, pattern)
        #cv2.waitKey(0)
        cv2.imwrite(r'.\output' + '\\' + title + '.png', pattern)
        #cv2.destroyWindow(title)