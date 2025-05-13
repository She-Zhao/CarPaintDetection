
# 🛠 漆面缺陷检测系统-重写版

## 🔍 0.为什么重写：
1. 系统所有执行逻辑堆叠，维护困难  
2. IO操作与算法处理叠加，资源浪费严重  
3. 启动程序繁琐，复现困难  
4. 缺少异常处理，调试困难  

## 📂 1.当前代码结构：
`feat/framework`分支下包含：
- `framework/`：重写后代码
- `source/`：原版代码
- `test/`：测试文件

**framework文件夹结构**：
```
core/              实际运行时的核心代码
├── client.py    → 原slave.py（从机启动）
├── server.py    → 原host.py（主机启动）
└── jiege.py     # 机械臂启动

data/                  代码运行用到的数据
├── patterns/          # 投影图像
└── main_point.txt     # 机械臂点位

engine/                核心检测算法的代码
├── pmd/               # 相位算法
├── yolo/              # 检测算法（计划添加pmd_run.py/detect_run.py）
└── 成像/              # 成像算法（@郑豪杰）

module/                系统运行需要的模块化代码
├── Camera.py          # 相机处理
├── Host.py            # 主机逻辑
└── Socket.py          # socket通信

output/                # 输出三类图像：
                       # 1. 原始图 2. 相位图 3. 检测结果

scripts/               # 启动脚本（待优化）
test/                  # 开发过程中用于测试的文件。
tools/                 # 开发过程中可能用到的工具类文件。
                       # 如保存机械臂位置的save_robot_pose.py等。
```

## 🧠 2.代码结构基本设计思路
1. **engine/**  
   - 核心算法（现有PMD/YOLO），每个子文件夹都是一个算法，
   - 计划每个文件夹中添加xxx_run.py调用当前目录中的算法，后续算法有改动的时候，只要对run.py文件调整即可
   - 后续添加：  
     - 豪杰成像算法  
     - 赵航机械臂规划算法  

2. **core/**  
   - 负责执行逻辑的代码放在core中，即core中的文件只负责调用，尽量保持简洁。
   - 若涉及到新的硬件/模块，将其放在module中
   - 示例：`client.py`调用`pmd_run.py`  

3. **core与engine的协作**  
   - 算法类执行通过子线程执行。即core中的文件，以模块类方法调用算法执行函数
   - 示例：pmd算法的处理放置在pmd/pmd_run.py，然后在client.py中直接调pmd_run.py
   - 减少通过文件传输中间变量的过程以及读写类操作

**执行流水线**：  
`相机采集图像——>成像模块，进行拼接和有效区域提取——>pmd_module，生成相位图——>detect_module，得到检测结果。`

## ⚙️ 3.运行
所有需要运行的文件均放置在core中，包括如下三个文件。
```bash
# 从机端
python core/client.py   # 原slave.py

# 主机端
python core/server.py   # 原host.py

# 机械臂
python core/jiege.py    # 完全未改动
```

## 🔜 4.后续优化
- client.py与pmd_module的集成优化
- client.py与成像部分（豪杰）的集成。client.py中拿到相机采集到的图像中。
- client.py和detect_module的集成，可能需要更新下检测的算法。
