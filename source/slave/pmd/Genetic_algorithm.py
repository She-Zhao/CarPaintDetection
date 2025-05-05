import math
import numpy as np
import random

'''
功能：
多变量遗传算法优化

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
def fitness(binary, start, precision):
    aa = binary_decode(binary, start, precision)
    th1 = aa[0]
    th2 = aa[1]
    th3 = aa[2]
    th4 = aa[3]
    th5 = aa[4]
    bb = math.cos(th1*th2*th3)*th4**math.sin(th5)+math.cos(th2**th1)
    if bb<0:
        fit = bb*(-1)
    else:
        fit = 0
    return fit
    ##########################################
    #区域提取+pmd+平均梯度计算得到gredient
    #fit = 1/gredient
################################################################################################选择
#generation为该代待选择的种群
def select(generation, population, start, precision):
    #计算所有个体的适应度
    all_fitness = []
    for binary in generation:
        fit = fitness(binary, start, precision)
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

def search(final_generation, start, precision):
    #计算所有个体的适应度，找到适应度最高的个体
    all_fitness = []
    for binary in final_generation:
        fit = fitness(binary, start, precision)
        all_fitness.append(fit)
    index = all_fitness.index(max(all_fitness))
    #解码，返回最优解
    return binary_decode(final_generation[index], start, precision)

#population为种群大小；generations为进化代数，cross_probabilit为交叉概率
#mutation_probability为变异概率，[start,end]为定义域，precision为精度
def genetic_cal(population, generations, cross_probability, mutation_probability, start, end, precision):
    #确定初始化种群（如图所示大致区间为[15,17]
    generation = initialization(population, start, end, precision, 5)
    for i in range(generations):
        #选择
        generation = select(generation, population, start, precision)
        #交叉
        generation = cross(generation, population, cross_probability, start, end, precision)
        #变异
        generation = mutation(generation, population, mutation_probability, start, end, precision)
    #搜索最优解，打印结果
    best = search(generation, start, precision)
    print(best)

if __name__ == "__main__":
    genetic_cal(80,400,0.7,0.05,0,5,1)
