"""
文件: main.py
功能: 系统主入口，协调主机控制流程
依赖: 
- opencv-python (cv2): 键盘事件监听
- pathlib: 路径解析
- module.HostControl: 主机控制核心模块

典型用法:
>>> python main.py ./projection_images/
或
>>> from core import main
>>> main('./projection_images/')
"""

# -*-coding:utf-8 -*-
import cv2
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
framework_root = current_file.parent.parent
sys.path.insert(0, str(framework_root))

from module import HostControl


def main(folder_path):
    """系统主控制流程，这里用于实现按键拍照，不涉及机械臂。
    
    工作流程:
    1. 初始化主机控制器
    2. 建立从机连接
    3. 进入交互式拍照循环
    4. 安全清理资源

    Args:
        folder_path (str): 投影图像目录路径，要求包含按数字命名的png文件

    键盘控制:
    - 任意键: 触发拍照流程
    - q键: 安全退出程序
    - Ctrl+C: 强制退出

    异常处理:
    - 捕获所有未处理异常并打印错误信息
    - 确保最终资源释放
    """    
    host = None             # 防止未成功建立连接时，finally中没有host报NameError的错误
    try:
        host = HostControl(folder_path)

        host.Socket_init()
        host.Projected_init()

        print("按任意键拍照，按q键退出...")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("正在退出程序...")
                break
            print(f"正在拍照（按下键: {chr(key)}）")
            host.Take_photo()

    except KeyboardInterrupt:           # 新增Ctrl+C捕获
        print("\n检测到强制退出!")
    except Exception as e:              # 新增异常捕获
        print(f"程序异常: {str(e)}")
    finally:
        if host:
            host.Disconnect()
        cv2.destroyAllWindows()         # 清理OpenCV窗口


if __name__ == '__main__':
    folder_path = r'D:\Project\CarPaintDetection\framework\data\patterns\nums10'
    main(folder_path)
