# run_visualizer.py
import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# 导入主窗口
from framework.gui.main_window import DefectVisualizer

if __name__ == "__main__":
    # 自动识别 output 路径 (假设在项目根目录的 output)
    local_output_path = Path(__file__).parent / "output"
    
    app = QApplication(sys.argv)
    
    # 高分屏适配
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    window = DefectVisualizer(data_root=local_output_path)
    window.showMaximized() 
    
    sys.exit(app.exec_())