import random
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
from concurrent.futures import ThreadPoolExecutor
import statistics





'''
功能：
遗传算法求解阈值

'''
'''
区域提取部分
'''
def cal_S(I):
    hist, bins = np.histogram(I.flatten(), bins=32)
    variance = statistics.variance(hist)
    return (math.log(variance,10)/8)**4
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

    # cv.imshow('mat', i01)
    # k = cv2.waitKey()
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
    # cv.imshow('mat', i01)
    # k = cv2.waitKey()
##################################################遍历求解
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

    # img_phase = compute_phase_cuda(photo_path,th1 = 2.3,th2 = 1.5,th3 = 3.5 ,th4 = 1,th5 = 1)
    # idx_phase = (i01 == 0)
    # img_phase_copy = img_phase
    # img_phase_copy[idx_phase] = 0
    idx_phase = (i01 == 255)
    return idx_phase
###################################################
'''
遗传算法部分
'''
#遗传算法部分
########################################################################################解码编码
#计算位数，[start,end]为取值范围，precision为精度，即小数点后的位数
def get_binary_bit(start, end, precision):
    numbers = (end - start) * pow(10, precision) + 1
    if int(math.log(numbers, 2)) == math.log(numbers, 2):
        return int(math.log(numbers, 2))
    else:
        return int(math.log(numbers, 2)) + 1
#单变量编码，decimal为未编码的十进制数列表,
def single_binary_encode(decimal, start, end, precision):
    bit = get_binary_bit(start, end, precision)
    #将十进制数转为对应的二进制编码
    binary = bin(int((decimal - start) * pow(10, precision)))
    #由于bin()生成的是0bxxxxx的形式，因此切片
    binary = str(binary)[2:]
    #补齐位数
    while len(binary) < bit:
        binary = '0' + binary
    #返回二进制编码
    return binary
#多变量编码
def binary_encode(delist,start,end,precision):
    binary = ''
    for i in range(len(delist)):
        binary = binary + single_binary_encode(delist[i],start,end,precision)
    return binary

#单变量解码，binary为二进制编码
def single_binary_decode(binary, start, precision):
    #将二进制编码转为标准形式0bxxxxx
    binary = '0b' + binary
    #将二进制编码转为十进制编码
    decimal = int(binary, 2)
    #将十进制编码转为对应的十进制数
    decimal = start + decimal / pow(10, precision)
    return decimal
#多变量解码
def binary_decode(binary, start, precision):
    bit = get_binary_bit(0,5,1)
    list = []
    start_flag = 0
    end_flag = start_flag+bit
    #将二进制编码转为标准形式0bxxxxx
    for i in range(5):
        aa = binary[start_flag:end_flag]
        decimal = single_binary_decode(aa,start,precision)
        list.append(decimal)
        start_flag = start_flag+bit
        end_flag = end_flag +bit
    return list
#########################################################################################################确定初始种群
#population为种群大小
def initialization(population, start, end, precision,variable_num):
    initialized = []
    for i in range(population):
        #随机生成指定范围和精度的小数，并转为二进制编码
        delist = []
        for j in range(variable_num):
            random_float = random.uniform(start, end)
            random_float_precision = round(random_float, precision)
            delist.append(random_float_precision)
        random_binary = binary_encode(delist, start, end, precision)
        initialized.append(random_binary)
    #返回初始化种群
    return initialized
#########################################################################################################################
#适应度函数，适应度的值越高，表示适应性越好，用Pmd后选定区域的平均梯度的倒数表示
def fitness(binary, start, precision,idx,photo_path,flag_generation):
    aa = binary_decode(binary, start, precision)
    th1 = aa[0]
    th2 = aa[1]
    th3 = aa[2]
    th4 = aa[3]
    th5 = aa[4]
    img_phase = compute_phase_cuda(photo_path,th1 ,th2 ,th3 ,th4,th5)
    gre = cal_gradient(img_phase[idx])
    # Sss = cal_S(img_phase[idx])
    
    if flag_generation <= 120:
        fit = 1/gre
    else:
        fit = 1/(1+math.exp(gre))
    return fit

###################################################
    ##########################################
    #区域提取+pmd+平均梯度计算得到gredient
    #fit = 1/gredient
################################################################################################选择
#generation为该代待选择的种群
def select(generation, population, start, precision,idx,photo_path,flag_generation):
    #计算所有个体的适应度
    all_fitness = []
    for binary in generation:
        fit = fitness(binary, start, precision,idx,photo_path,flag_generation)
        all_fitness.append(fit)
    fitness_array = np.array(all_fitness)
    #将适应度归一化，作为概率进行选择
    fitness_array = fitness_array / fitness_array.sum()
    selected_index = np.random.choice(list(range(population)), size=population, p=fitness_array)
    selected_index = selected_index.tolist()
    #将选择到的个体加入选择后的种群
    selected = []
    for index in selected_index:
        selected.append(generation[index])
    #返回被选择之后的种群
    return selected
######################################################################################################交叉
#判断是否在范围内
def in_range(binary, start, end, precision):
    flag = True
    delist = binary_decode(binary,start,precision)
    for i in range(len(delist)):
        if start <= delist[i] <= end:
            flag = flag & True
        else:
            flag = flag & False
    return flag
#selected为被选择之后的种群，probability为交叉概率
def cross(selected, population, probability, start, end, precision):
    crossed = selected[:]
    bit = get_binary_bit(start, end, precision)*5
    #numbers为进行交叉的次数
    numbers = population * probability
    count = 0
    i = 0
    while i < population - 1 and count < numbers:
        #随机选取分割点
        position = random.randrange(1, bit)
        #将两个父二进制编码分别截成两个部分
        binary11 = selected[i][:position]
        binary12 = selected[i][position:]
        binary21 = selected[i + 1][:position]
        binary22 = selected[i + 1][position:]
        #将二进制编码切片重组形成新的两个子二进制编码
        binary1 = binary11 + binary22
        binary2 = binary21 + binary12
        #判断新生成的二进制编码是否在自定义范围内，在则加入交叉之后的种群；否则，重新交叉
        if in_range(binary1, start, end, precision) and in_range(binary2, start, end, precision):
            crossed[i] = binary1
            crossed[i + 1] = binary2
            count += 1
            i += 2
    #返回交叉之后的种群
    return crossed

def reverse(string, position):
    string = list(string)
    if string[position] == '0':
        string[position] = '1'
    else:
        string[position] = '0'
    return ''.join(string)

#crossed为交叉之后的种群，probability为变异的概率
def mutation(crossed, population, probability, start, end, precision):
    mutated = crossed[:]
    bit = get_binary_bit(start, end, precision)*5
    for i in range(population):
        #随机生成一个0-1之间的数，判断该二进制编码是否突变
        whether_mutated = True if random.random() < probability else False
        if whether_mutated:
            #随机生成一个变异位，将该位进行取反
            position = random.randrange(0, bit)
            mutated_binary = reverse(crossed[i], position)
            #若生成的新二进制编码不在定义域内，则重复生成直至符合条件
            while not in_range(mutated_binary, start, end, precision):
                position = random.randrange(0, bit)
                mutated_binary = reverse(crossed[i], position)
            mutated[i] = mutated_binary
    #返回变异后的种群
    return mutated
#############################################################在最后一代种群中搜索最优解
#final_generation为最后一代种群

def search(final_generation, start, precision,idx,photo_path,flag_generation):
    #计算所有个体的适应度，找到适应度最高的个体
    flag_generation = 1
    all_fitness = []
    for binary in final_generation:
        fit = fitness(binary, start, precision,idx,photo_path,flag_generation)
        all_fitness.append(fit)
    index = all_fitness.index(max(all_fitness))
    #解码，返回最优解
    return binary_decode(final_generation[index], start, precision)

#population为种群大小；generations为进化代数，cross_probabilit为交叉概率
#mutation_probability为变异概率，[start,end]为定义域，precision为精度
def genetic_cal(population, generations, cross_probability, mutation_probability, start, end, precision,idx,photo_path):
    #确定初始化种群（如图所示大致区间为[15,17]
    generation = initialization(population, start, end, precision, 5)
    for i in range(generations):
        #选择
        generation = select(generation, population, start, precision,idx,photo_path,flag_generation=i)
        #交叉
        generation = cross(generation, population, cross_probability, start, end, precision)
        #变异
        generation = mutation(generation, population, mutation_probability, start, end, precision)

    #搜索最优解，打印结果
    best = search(generation, start, precision,idx,photo_path,flag_generation=1)
    return best
    print(best)

def final_cal(photo_path):
    idx = area_cal(photo_path)
    best = genetic_cal(40, 250, 0.7, 0.08, 0.1, 6, 1, idx, photo_path)
    print_name = photo_path.split('cam',1)[1][1:]
    print('',print_name,best)


if __name__ == "__main__":
    
    begin = time.time()
    print('0411_2')

    datapath = r'D:\zhj\0411\pic\cam'
    files = os.listdir(datapath)
    with ThreadPoolExecutor(max_workers=10) as executor:  # max_workers指定每次最多处理几组照片
        # 利用线程池并行执行compute_phase1函数
        executor.map(final_cal, [os.path.join(datapath, file) for file in files])
    end = time.time()
    t = end - begin
    print("总时间：", t)

