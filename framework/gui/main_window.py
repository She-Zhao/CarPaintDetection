# framework/gui/main_window.py

import json
import cv2
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QListWidget, QLabel, QGraphicsScene, QGraphicsPixmapItem, 
                             QSplitter, QComboBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QAbstractItemView, QGroupBox, QSizePolicy, QPushButton)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QBrush, QColor, QFont

# 引入拆分后的模块
from framework.gui.styles import DARK_STYLESHEET, DEFECT_CLASSES
from framework.gui.widgets import ZoomableGraphicsView, InteractiveDefectBox
from .threads import DataReceiverThread, RobotSimulationThread

class DefectVisualizer(QMainWindow):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = Path(data_root)
        
        # 线程引用
        self.receiver_thread = None
        self.robot_thread = None
        
        # 运行时数据
        self.current_pos_dir = None
        self.current_img_np = None
        self.defects_data = []
        self.box_items = []
        
        self.setStyleSheet(DARK_STYLESHEET)
        self.init_ui()
        self.load_pos_list()
        
        # 自动启动接收服务
        self.start_receiver_service()

        if self.pos_list_widget.count() > 0:
            self.pos_list_widget.setCurrentRow(0)

    def init_ui(self):
        self.setWindowTitle("漆面缺陷智能检测系统 - 工业控制台")
        self.resize(1600, 950)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # === 左侧面板 ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        lbl_title = QLabel("POSITIONS")
        lbl_title.setObjectName("TitleLabel")
        left_layout.addWidget(lbl_title)

        self.pos_list_widget = QListWidget()
        self.pos_list_widget.currentRowChanged.connect(self.on_pos_selected)
        left_layout.addWidget(self.pos_list_widget)
        
        # 控制按钮
        btn_layout = QVBoxLayout()
        self.btn_start = QPushButton("开始自动检测")
        self.btn_start.clicked.connect(self.toggle_robot_task)
        self.btn_start.setMinimumHeight(40)
        btn_layout.addWidget(self.btn_start)
        left_layout.addLayout(btn_layout)

        left_layout.addWidget(QLabel("Tip: A/D 切图, 滚轮缩放"))

        # === 中间面板 ===
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)

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

        # === 右侧面板 ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

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

    # === 业务逻辑方法 (保持原样) ===
    def start_receiver_service(self):
        self.receiver_thread = DataReceiverThread(self.data_root)
        self.receiver_thread.data_received.connect(self.on_new_data_received)
        self.receiver_thread.log_message.connect(lambda msg: print(msg)) 
        self.receiver_thread.start()

    def toggle_robot_task(self):
        if self.robot_thread and self.robot_thread.isRunning():
            self.robot_thread.stop()
            self.btn_start.setText("正在停止...")
            self.btn_start.setEnabled(False)
        else:
            # TODO: 修改为实际路径
            project_folder = r'D:\Project\CarPaintDetection\framework\cfg\patterns\nums10' 
            self.robot_thread = RobotSimulationThread(project_folder)
            self.robot_thread.log_message.connect(self.lbl_pos_info.setText)
            self.robot_thread.task_finished.connect(self.on_robot_task_finished)
            self.robot_thread.start()
            self.btn_start.setText("停止检测")
            self.btn_start.setStyleSheet("background-color: #d81e06; color: white;")

    def on_robot_task_finished(self):
        self.btn_start.setText("开始自动检测")
        self.btn_start.setStyleSheet("")
        self.btn_start.setEnabled(True)
        self.lbl_pos_info.setText("检测任务结束")

    def on_new_data_received(self, pos_id, is_last):
        dir_name = f"pos{pos_id}"
        items = self.pos_list_widget.findItems(dir_name, Qt.MatchExactly)
        if not items:
            self.pos_list_widget.addItem(dir_name)
            self.pos_list_widget.scrollToBottom()
        
        if is_last:
            items = self.pos_list_widget.findItems(dir_name, Qt.MatchExactly)
            if items:
                self.pos_list_widget.setCurrentItem(items[0])
                self.on_pos_selected(self.pos_list_widget.row(items[0]))

    # ... (此处省略了 keyPressEvent, showEvent, load_pos_list 等常规GUI逻辑) ...
    # ... 请将原 visualizer.py 中剩余的方法直接复制到这里 ...
    # 包含: keyPressEvent, showEvent, resizeEvent, fit_image, load_pos_list, 
    # on_pos_selected, load_defects_data, update_stats_table, refresh_view, 
    # draw_defects, update_defects_table, on_table_row_clicked, interact_select_defect, update_roi_view
    
    # -------------------------------------------------------------
    # 注意：为了代码完整性，请务必把上面省略的方法从你原来的 visualizer.py 
    # 复制到这个类下面。逻辑不需要任何修改。
    # -------------------------------------------------------------