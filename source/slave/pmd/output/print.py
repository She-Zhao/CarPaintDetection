from concurrent.futures import ThreadPoolExecutor
import os

path = r'D:\zhj\0409\pic'
files = os.listdir(path)
def print_file(aa):
    file = aa.split('pic',1)[1][1:]
    print(file)
with ThreadPoolExecutor(max_workers=2) as executor:  # max_workers指定每次最多处理几组照片
    # 利用线程池并行执行compute_phase1函数
    executor.map(print_file, [os.path.join(path, file) for file in files])