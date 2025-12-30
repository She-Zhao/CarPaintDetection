# framework/gui/main_window.py

import json
import cv2
import numpy as np
import datetime
from pathlib import Path

from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QListWidget, QLabel, QGraphicsScene, QGraphicsPixmapItem, 
                             QSplitter, QComboBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QAbstractItemView, QGroupBox, QSizePolicy, 
                             QPushButton, QTabWidget, QTextEdit, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QFileSystemWatcher
from PyQt5.QtGui import QPixmap, QImage, QBrush, QColor, QFont

# 引入模块
from framework.gui.styles import DARK_STYLESHEET, DEFECT_CLASSES
from framework.gui.widgets import ZoomableGraphicsView, InteractiveDefectBox
from .threads import DataReceiverThread, RobotSimulationThread


# =============================================================================
# Tab 1: 离线查看模式 (OfflineTab)
# =============================================================================
class OfflineTab(QWidget):
    """
    复用原本的主界面逻辑，用于事后回溯查看数据
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_root = None 
        self.watcher = None # 文件系统监控器
        self.current_pos_dir = None
        self.current_img_np = None
        self.defects_data = []
        self.box_items = []
        
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # === 左侧列表 ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0,0,0,0)
        
        left_layout.addWidget(QLabel("POSITIONS"))
        self.pos_list_widget = QListWidget()
        self.pos_list_widget.currentRowChanged.connect(self.on_pos_selected)
        left_layout.addWidget(self.pos_list_widget)
        left_layout.addWidget(QLabel("Tip: A/D 切图, 滚轮缩放"))

        # === 中间视图 ===
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0,0,0,0)

        tool_layout = QHBoxLayout()
        self.view_selector = QComboBox()
        self.view_selector.addItems(["Phase Map (相位图)", "Stitched Image (拼接图)", "Raw Image (原始图)"])
        self.view_selector.setMinimumWidth(200)
        self.view_selector.currentIndexChanged.connect(self.refresh_view)
        
        self.lbl_pos_info = QLabel("Ready")
        self.lbl_pos_info.setStyleSheet("color: #00aaff; font-weight: bold;")

        tool_layout.addWidget(self.view_selector)
        tool_layout.addStretch()
        tool_layout.addWidget(self.lbl_pos_info)
        
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QBrush(QColor("#1e1e1e")))
        self.view = ZoomableGraphicsView(self.scene)
        
        center_layout.addLayout(tool_layout)
        center_layout.addWidget(self.view)

        # === 右侧详情 ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0,0,0,0)

        # 统计
        stats_group = QGroupBox("STATISTICS")
        stats_layout = QVBoxLayout()
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setFixedHeight(80) 
        self.stats_table.setShowGrid(False) 
        self.stats_table.setStyleSheet("border: none; background-color: transparent;")
        stats_layout.addWidget(self.stats_table)
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group, stretch=1)

        # 列表
        list_group = QGroupBox("DEFECT LIST")
        list_layout = QVBoxLayout()
        self.defect_table = QTableWidget()
        self.defect_table.setColumnCount(3)
        self.defect_table.setHorizontalHeaderLabels(["ID", "类型", "Conf"])
        self.defect_table.verticalHeader().setVisible(False)
        self.defect_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.defect_table.cellClicked.connect(self.on_table_row_clicked)
        self.defect_table.setColumnWidth(0, 40)
        self.defect_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        list_layout.addWidget(self.defect_table)
        list_group.setLayout(list_layout)
        right_layout.addWidget(list_group, stretch=2)

        # 特写
        roi_group = QGroupBox("INSPECTION")
        roi_layout = QVBoxLayout()
        self.roi_label = QLabel("No Selection")
        self.roi_label.setAlignment(Qt.AlignCenter)
        self.roi_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.roi_label.setStyleSheet("background-color: #000; border: 1px solid #555;")
        roi_layout.addWidget(self.roi_label)
        roi_group.setLayout(roi_layout)
        right_layout.addWidget(roi_group, stretch=3)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 5)
        
        main_layout.addWidget(splitter)

    # --- 业务逻辑 ---
    def set_data_root(self, data_root):
        self.data_root = data_root
        
        # [修改点1] 初始化文件监控器
        if self.watcher:
            self.watcher.removePaths(self.watcher.directories())
        self.watcher = QFileSystemWatcher()
        if self.data_root.exists():
            self.watcher.addPath(str(self.data_root))
            # 当目录改变（如删除/新建文件夹）时，触发刷新
            self.watcher.directoryChanged.connect(self.load_pos_list)
        
        self.load_pos_list()

    def load_pos_list(self):
        """加载点位列表，支持被 Watcher 触发"""
        current_row = self.pos_list_widget.currentRow()
        self.pos_list_widget.clear()
        
        if not self.data_root or not self.data_root.exists(): return
        
        # 重新扫描
        dirs = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("pos")]
        dirs.sort(key=lambda x: int(x.name.replace("pos", "")) if x.name.replace("pos", "").isdigit() else 0)
        
        for d in dirs:
            self.pos_list_widget.addItem(d.name)
            
        # 尝试保持之前的选中状态，或者选中最新
        if self.pos_list_widget.count() > 0:
            if current_row >= 0 and current_row < self.pos_list_widget.count():
                self.pos_list_widget.setCurrentRow(current_row)
            else:
                self.pos_list_widget.setCurrentRow(self.pos_list_widget.count() - 1)

    def on_pos_selected(self, index):
        if index < 0: return
        item = self.pos_list_widget.item(index)
        self.current_pos_dir = self.data_root / item.text()
        self.lbl_pos_info.setText(f"CURRENT: {item.text()}")
        
        self.load_defects_data()
        self.update_stats_table()
        self.update_defects_table()
        self.refresh_view()

    def refresh_view(self):
        if not self.current_pos_dir: return
        mode = self.view_selector.currentText()
        img_name = "phase_1.png"
        if "Processed" in mode: img_name = "processed_5.png"
        elif "Raw" in mode: img_name = "cam1_sin0.png"
        
        img_path = self.current_pos_dir / img_name
        self.scene.clear()
        self.box_items = []

        if img_path.exists():
            self.current_img_np = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.current_img_np is None: return
            
            img_rgb = cv2.cvtColor(self.current_img_np, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.img_item = QGraphicsPixmapItem(QPixmap.fromImage(q_img))
            self.scene.addItem(self.img_item)
            self.draw_defects()
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        else:
            self.scene.addText(f"Image Not Found: {img_name}", QFont("Arial", 20)).setDefaultTextColor(Qt.red)

    def load_defects_data(self):
        self.defects_data = []
        json_path = self.current_pos_dir / "defects.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    raw_data = json.load(f)
                if len(raw_data) > 0 and isinstance(raw_data[0], list):
                     self.defects_data = [d for sublist in raw_data for d in sublist] if isinstance(raw_data[0][0], list) else [d for sublist in raw_data for d in sublist]
                else:
                    self.defects_data = raw_data
            except Exception: pass

    def update_stats_table(self):
        stats = {}
        for defect in self.defects_data:
            cls_id = int(defect[0])
            stats[cls_id] = stats.get(cls_id, 0) + 1
        
        self.stats_table.setRowCount(len(stats))
        sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        for i, (cls_id, count) in enumerate(sorted_stats):
            name = DEFECT_CLASSES.get(cls_id, f"Type {cls_id}")
            self.stats_table.setItem(i, 0, QTableWidgetItem(name))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(count)))

    def draw_defects(self):
        font = QFont("Arial", 10, QFont.Bold)
        for i, defect in enumerate(self.defects_data):
            cls_id, x, y, w, h, conf = defect
            box = InteractiveDefectBox(x, y, w, h, i, self.interact_select_defect)
            name = DEFECT_CLASSES.get(int(cls_id), "Unknown")
            box.setToolTip(f"ID:{i}\nType: {name}\nConf: {conf:.2f}") 
            self.scene.addItem(box)
            self.box_items.append(box)
            text = self.scene.addText(f"#{i}", font)
            text.setDefaultTextColor(QColor(255, 50, 50))
            text.setPos(x, y - 25)

    def update_defects_table(self):
        self.defect_table.setRowCount(len(self.defects_data))
        for i, defect in enumerate(self.defects_data):
            cls_id, _, _, _, _, conf = defect
            name = DEFECT_CLASSES.get(int(cls_id), str(cls_id))
            item_id = QTableWidgetItem(str(i)); item_id.setTextAlignment(Qt.AlignCenter)
            item_type = QTableWidgetItem(name)
            item_conf = QTableWidgetItem(f"{conf:.2f}"); item_conf.setTextAlignment(Qt.AlignCenter)
            self.defect_table.setItem(i, 0, item_id)
            self.defect_table.setItem(i, 1, item_type)
            self.defect_table.setItem(i, 2, item_conf)

    def on_table_row_clicked(self, row, col):
        self.interact_select_defect(row)

    def interact_select_defect(self, index):
        if index < 0 or index >= len(self.defects_data): return
        self.defect_table.selectRow(index)
        self.defect_table.scrollToItem(self.defect_table.item(index, 0))
        for i, box in enumerate(self.box_items):
            box.set_highlight(i == index)
        self.update_roi_view(index)

    def update_roi_view(self, index):
        if self.current_img_np is None: return
        defect = self.defects_data[index]
        x, y, w, h = map(int, defect[1:5])
        pad = 50 
        img_h, img_w, _ = self.current_img_np.shape
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_w, x + w + pad), min(img_h, y + h + pad)
        roi = self.current_img_np[y1:y2, x1:x2]
        if roi.size > 0:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            h_roi, w_roi, ch = roi.shape
            q_roi = QImage(roi.data, w_roi, h_roi, ch * w_roi, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_roi)
            self.roi_label.setPixmap(pixmap.scaled(self.roi_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# =============================================================================
# Tab 2: 在线检测模式 (OnlineTab)
# =============================================================================
class OnlineTab(QWidget):
    """
    改进版工业监控：
    1. 支持缺陷叠加显示 (BBox)
    2. 支持视角切换 (Phase/Stitched/Raw)
    3. 右侧新增详细信息看板
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_root = None
        self.current_pos_dir = None
        self.defects_data = [] # 存储当前缺陷
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # === 左侧控制区 (Controls) ===
        left_layout = QVBoxLayout()
        
        self.btn_start = QPushButton("启动自动检测")
        self.btn_start.setMinimumHeight(60)
        self.btn_start.setStyleSheet("""
            QPushButton { font-size: 14pt; background-color: #0078d7; color: white; border-radius: 8px; }
            QPushButton:hover { background-color: #1084e3; }
            QPushButton:disabled { background-color: #555; }
        """)
        left_layout.addWidget(self.btn_start)
        
        # 状态指示灯
        self.lbl_status = QLabel("● 系统就绪")
        self.lbl_status.setStyleSheet("color: #00ff00; font-size: 12pt; font-weight: bold; margin-top: 20px;")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.lbl_status)
        
        left_layout.addStretch()
        
        # === 中间实时画面 (Monitor) ===
        center_layout = QVBoxLayout()
        
        # [修改点2] 增加视角切换下拉框
        tool_layout = QHBoxLayout()
        tool_layout.addWidget(QLabel("Real-time Monitor (实时监控)"))
        
        self.view_selector = QComboBox()
        self.view_selector.addItems(["Phase Map (相位图)", "Stitched Image (拼接图)", "Raw Image (原始图)"])
        self.view_selector.currentIndexChanged.connect(self.refresh_view) # 切换时触发重绘
        tool_layout.addWidget(self.view_selector)
        tool_layout.addStretch()
        
        center_layout.addLayout(tool_layout)
        
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QBrush(QColor("#000000")))
        self.view = ZoomableGraphicsView(self.scene)
        center_layout.addWidget(self.view)
        
        # === 右侧信息区 (Dashboard + Log) ===
        right_layout = QVBoxLayout()
        
        # [修改点3] 新增检测看板 (Dashboard)
        dash_group = QGroupBox("DASHBOARD (看板)")
        dash_layout = QVBoxLayout()
        
        # 显示当前点位
        self.lbl_current_pos = QLabel("Position: --")
        self.lbl_current_pos.setStyleSheet("font-size: 16pt; font-weight: bold; color: #00aaff; margin-bottom: 10px;")
        dash_layout.addWidget(self.lbl_current_pos)
        
        # 显示缺陷统计表格
        self.stat_table = QTableWidget()
        self.stat_table.setColumnCount(2)
        self.stat_table.setHorizontalHeaderLabels(["Defect Type", "Count"])
        self.stat_table.verticalHeader().setVisible(False)
        self.stat_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stat_table.setStyleSheet("border: none; background-color: #2b2b2b;")
        self.stat_table.setFixedHeight(150)
        dash_layout.addWidget(self.stat_table)
        
        dash_group.setLayout(dash_layout)
        right_layout.addWidget(dash_group)
        
        # 日志区
        right_layout.addWidget(QLabel("System Log"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #222; color: #00ff00; font-family: Consolas; font-size: 9pt;")
        self.log_text.setFixedWidth(350)
        right_layout.addWidget(self.log_text)

        # 布局组合
        main_layout.addLayout(left_layout)
        main_layout.addLayout(center_layout, stretch=1)
        main_layout.addLayout(right_layout)

    @pyqtSlot(str)
    def append_log(self, msg):
        """追加日志"""
        timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
        self.log_text.append(f"{timestamp} {msg}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @pyqtSlot(int)
    def update_monitor(self, pos_id, data_root):
        """收到新数据时调用"""
        self.current_pos_dir = data_root / f"pos{pos_id}"
        
        # 1. 更新看板信息
        self.lbl_current_pos.setText(f"Position: {pos_id}")
        self.load_defects_data() # 加载 JSON
        self.update_stat_panel() # 更新统计表
        
        # 2. 刷新图像和缺陷框
        self.refresh_view()
        
        self.append_log(f"-> 画面已更新: Pos {pos_id}")

    def load_defects_data(self):
        """加载当前点位的缺陷数据"""
        self.defects_data = []
        if not self.current_pos_dir: return
        json_path = self.current_pos_dir / "defects.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    raw_data = json.load(f)
                # 兼容不同格式
                if len(raw_data) > 0 and isinstance(raw_data[0], list):
                     self.defects_data = [d for sublist in raw_data for d in sublist] if isinstance(raw_data[0][0], list) else [d for sublist in raw_data for d in sublist]
                else:
                    self.defects_data = raw_data
            except Exception: pass

    def update_stat_panel(self):
        """更新右侧统计表格"""
        stats = {}
        for defect in self.defects_data:
            cls_id = int(defect[0])
            stats[cls_id] = stats.get(cls_id, 0) + 1
            
        self.stat_table.setRowCount(len(stats))
        sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        for i, (cls_id, count) in enumerate(sorted_stats):
            name = DEFECT_CLASSES.get(cls_id, f"Type {cls_id}")
            self.stat_table.setItem(i, 0, QTableWidgetItem(name))
            item_count = QTableWidgetItem(str(count))
            item_count.setTextAlignment(Qt.AlignCenter)
            self.stat_table.setItem(i, 1, item_count)

    def refresh_view(self):
        """根据当前选择的视角重绘图像和缺陷框"""
        if not self.current_pos_dir: return
        
        mode = self.view_selector.currentText()
        img_name = "phase_1.png"
        if "Processed" in mode: img_name = "processed_5.png"
        elif "Raw" in mode: img_name = "cam1_sin0.png"
        
        img_path = self.current_pos_dir / img_name
        self.scene.clear()
        
        if img_path.exists():
            # 加载图像
            img_np = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_np is not None:
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(q_img)))
                
                # [修改点2] 绘制缺陷框 (复用逻辑)
                self.draw_defects_on_monitor()
                
                self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        else:
            self.scene.addText(f"Waiting for Image...", QFont("Arial", 20)).setDefaultTextColor(Qt.white)

    def draw_defects_on_monitor(self):
        """在实时画面上绘制缺陷框"""
        font = QFont("Arial", 10, QFont.Bold)
        for i, defect in enumerate(self.defects_data):
            cls_id, x, y, w, h, conf = defect
            # 在线模式下，框可以设为不可交互，或者保持交互但不做跳转
            # 这里简单起见，使用 InteractiveDefectBox 但回调设为空 lambda
            box = InteractiveDefectBox(x, y, w, h, i, lambda idx: None)
            
            name = DEFECT_CLASSES.get(int(cls_id), "Unknown")
            box.setToolTip(f"Type: {name}\nConf: {conf:.2f}") 
            self.scene.addItem(box)
            
            # 绘制标签
            text = self.scene.addText(f"{name}", font)
            text.setDefaultTextColor(QColor(0, 255, 0)) # 在线模式用绿色字
            text.setPos(x, y - 25)


# =============================================================================
# 主窗口：DefectVisualizer (整合 Tabs)
# =============================================================================
class DefectVisualizer(QMainWindow):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = Path(data_root)
        self.setWindowTitle("漆面缺陷智能检测系统 - 工业控制台")
        self.resize(1600, 950)
        self.setStyleSheet(DARK_STYLESHEET)
        
        # 线程引用
        self.receiver_thread = None
        self.robot_thread = None
        
        self.init_ui()
        self.start_receiver_service()

    def init_ui(self):
        # 创建 Tab Widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: #aaa; padding: 10px 20px; }
            QTabBar::tab:selected { background: #444; color: white; border-bottom: 2px solid #0078d7; }
        """)
        
        # 实例化两个页面
        self.tab_offline = OfflineTab()
        self.tab_online = OnlineTab()
        
        # 传递数据路径
        self.tab_offline.set_data_root(self.data_root)
        self.tab_online.data_root = self.data_root 
        
        # 绑定 Online Tab 的按钮事件到主窗口的逻辑
        self.tab_online.btn_start.clicked.connect(self.toggle_robot_task)
        
        self.tabs.addTab(self.tab_offline, "📚 离线数据分析 (Offline)")
        self.tabs.addTab(self.tab_online, "🚀 在线实时监控 (Online)")
        
        self.setCentralWidget(self.tabs)

    # --- 线程与业务逻辑 ---
    
    def start_receiver_service(self):
        self.receiver_thread = DataReceiverThread(self.data_root)
        self.receiver_thread.data_received.connect(self.on_new_data_received)
        # 将底层接收日志直接打到在线 Tab 的日志框里
        self.receiver_thread.log_message.connect(self.tab_online.append_log)
        self.receiver_thread.start()

    def toggle_robot_task(self):
        """控制 Robot 线程的启停"""
        if self.robot_thread and self.robot_thread.isRunning():
            self.robot_thread.stop()
            self.tab_online.btn_start.setText("正在停止...")
            self.tab_online.btn_start.setEnabled(False)
        else:
            # TODO: 修改为实际路径
            project_folder = r'D:\Project\CarPaintDetection\framework\cfg\patterns\pmd' 
            self.robot_thread = RobotSimulationThread(project_folder)
            
            # 绑定日志
            self.robot_thread.log_message.connect(self.tab_online.append_log)
            self.robot_thread.task_finished.connect(self.on_robot_task_finished)
            
            self.robot_thread.start()
            
            # 更新 UI 状态
            self.tab_online.btn_start.setText("🛑 停止检测")
            self.tab_online.btn_start.setStyleSheet("background-color: #d81e06; color: white; font-size: 14pt; border-radius: 8px;")
            self.tab_online.lbl_status.setText("● 正在运行")
            self.tab_online.lbl_status.setStyleSheet("color: #00ff00; font-size: 12pt; font-weight: bold; margin-top: 20px;")
            
            self.tabs.setCurrentWidget(self.tab_online)

    def on_robot_task_finished(self):
        self.tab_online.btn_start.setText("启动自动检测")
        self.tab_online.btn_start.setStyleSheet("background-color: #0078d7; color: white; font-size: 14pt; border-radius: 8px;")
        self.tab_online.btn_start.setEnabled(True)
        self.tab_online.lbl_status.setText("● 任务结束")
        self.tab_online.lbl_status.setStyleSheet("color: #ffa500; font-size: 12pt; font-weight: bold; margin-top: 20px;")
        self.tab_online.append_log(">>> 检测任务已结束")

    def on_new_data_received(self, pos_id, is_last):
        """当后台收到新数据时"""
        # 1. 触发离线列表刷新 (现在有了 Watcher，这一步其实是双重保险)
        self.tab_offline.load_pos_list()
        
        # 2. 如果当前正在 Online 模式，且收到了该组最后一张图(is_last=True)，则刷新监控画面
        if is_last:
            self.tab_online.update_monitor(pos_id, self.data_root)