# run_visualizer.py
import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# 导入主窗口
from framework.gui.main_window import DefectVisualizer

if __name__ == "__main__":
    local_output_path = Path(__file__).parent.parent.parent / "received_output"
    
    app = QApplication(sys.argv)
    
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    window = DefectVisualizer(data_root=local_output_path)
    
    # === [新增] 强制窗口在主屏幕启动 ===
    # 获取主屏幕 (索引0) 的几何信息
    primary_screen = app.desktop().screenGeometry(0)
    
    # 将窗口移动到主屏幕的左上角区域 (稍微偏移一点)，确保它属于主屏
    window.move(primary_screen.x() + 50, primary_screen.y() + 50)
    # =================================
    
    window.showMaximized() 
    
    sys.exit(app.exec_())
