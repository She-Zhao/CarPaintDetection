"""
sever_capture.py - 主机拍照控制模块

该模块负责控制主机拍照流程，与从机配合完成图像采集任务。不需要机械臂的参与。

功能说明:
    - 初始化主机相机和投影仪
    - 监听键盘输入触发拍照
    - 安全释放资源

典型用法:
    >>> folder_path = "path/to/patterns"
    >>> main(folder_path)

注意事项:
    - 需先启动从机程序
    - 按任意键拍照，按q键退出
    - 支持Ctrl+C强制退出

最后一次修改：2025/05/05
"""

# -*-coding:utf-8 -*-
import cv2
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
framework_root = current_file.parent.parent
sys.path.insert(0, str(framework_root))

from module import Host


def main(folder_path):
    host = None             # 防止未成功建立连接时，finally中没有host报NameError的错误
    try:
        host = Host(folder_path)

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

    except KeyboardInterrupt:  # 新增Ctrl+C捕获
        print("\n检测到强制退出!")
    except Exception as e:     # 新增异常捕获
        print(f"程序异常: {str(e)}")
    finally:
        if host:
            host.Disconnect()
        cv2.destroyAllWindows()  # 清理OpenCV窗口


if __name__ == '__main__':
    folder_path = r'D:\Project\CarPaintDetection\framework\data\patterns\nums10'
    main(folder_path)
