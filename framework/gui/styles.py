# framework/gui/styles.py

# 缺陷类别映射表
DEFECT_CLASSES = {
    0: "Breakage (破损)",
    1: "Inclusion (夹杂)",
    2: "Scratch (划痕)",
    3: "Crater (缩孔)",
    4: "Run (流挂)",
    5: "Bulge (凸起)",
    6: "Condensate (冷凝)"
}

# 全局样式表 (Dark Theme)
DARK_STYLESHEET = """
QMainWindow { background-color: #2b2b2b; }
QWidget { color: #e0e0e0; font-family: "Segoe UI", "Microsoft YaHei"; font-size: 10pt; }
QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; font-weight: bold; background-color: #333; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: #aaa; }
QListWidget, QTableWidget { background-color: #3c3f41; border: 1px solid #555; border-radius: 4px; gridline-color: #444; }
QListWidget::item:selected, QTableWidget::item:selected { background-color: #0078d7; color: white; }
QListWidget::item:hover, QTableWidget::item:hover { background-color: #4c5052; }
QHeaderView::section { background-color: #444; color: #ccc; padding: 4px; border: none; border-bottom: 1px solid #555; }
QComboBox { background-color: #3c3f41; color: #ffffff; border: 1px solid #555; padding: 5px; border-radius: 4px; }
QComboBox:hover { border: 1px solid #0078d7; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView { background-color: #3c3f41; color: #ffffff; selection-background-color: #0078d7; border: 1px solid #555; }
QScrollBar:vertical { background: #2b2b2b; width: 10px; }
QScrollBar::handle:vertical { background: #555; border-radius: 5px; }
QPushButton { background-color: #0078d7; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; }
QPushButton:hover { background-color: #1084e3; }
QPushButton:pressed { background-color: #0063b1; }
QPushButton:disabled { background-color: #555; color: #888; }
QLabel#TitleLabel { font-size: 12pt; font-weight: bold; color: #00aaff; }
"""