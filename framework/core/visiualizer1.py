import sys
import json
import cv2
import numpy as np
import socket
import time
import struct
from pathlib import Path

# PyQt5 相关
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QListWidget, QLabel, QGraphicsView, 
                             QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
                             QFrame, QSplitter, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QAbstractItemView, QGroupBox, QSizePolicy, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QBrush, QPainter, QFont

# 引入你的模块
from framework.module import HostControl
from framework.module.transfer import DataProtocol

# --- 全局样式表 (Dark Theme) 保持不变 ---
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

DEFECT_CLASSES = {
    0: "Breakage (破损)", 1: "Inclusion (夹杂)", 2: "Scratch (划痕)",
    3: "Crater (缩孔)", 4: "Run (流挂)", 5: "Bulge (凸起)", 6: "Condensate (冷凝)"
}

# =============================================================================
# 线程 1: 数据接收线程 (前身是 host_debug.py)
# =============================================================================
class DataReceiverThread(QThread):
    # 信号：通知 UI 有新数据包到达 (pos_id, is_last_of_batch)
    data_received = pyqtSignal(int, bool)
    log_message = pyqtSignal(str) # 发送日志到 UI

    def __init__(self, output_root: Path, port=4097):
        super().__init__()
        self.output_root = output_root
        self.port = port
        self.is_running = True
        self.server_socket = None

    def run(self):
        HOST = '0.0.0.0'
        self.output_root.mkdir(exist_ok=True, parents=True)
        
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 允许端口复用，防止重启程序报错
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((HOST, self.port))
            self.server_socket.listen(1)
            self.log_message.emit(f"[Receiver] 监听数据端口 {self.port}...")

            while self.is_running:
                # 设置 accept 超时，以便能响应 stop()
                self.server_socket.settimeout(1.0)
                try:
                    conn, addr = self.server_socket.accept()
                except socket.timeout:
                    continue
                except Exception as e:
                    self.log_message.emit(f"[Receiver] Accept Error: {e}")
                    continue

                self.log_message.emit(f"[Receiver] 从机已连接: {addr}")
                conn.settimeout(None) # 恢复阻塞模式或设置较长超时

                try:
                    while self.is_running:
                        # 1. 解包数据
                        try:
                            image, meta_data = DataProtocol.unpack_data(conn)
                        except (ConnectionError, struct.error):
                            self.log_message.emit("[Receiver] 从机断开连接")
                            break # 跳出内层循环，重新 accept

                        # 2. 解析元数据
                        if isinstance(meta_data, list): continue # 忽略旧版

                        pos_id = meta_data.get("pos_id", 0)
                        filename = meta_data.get("filename", f"unknown_{time.time()}.png")
                        is_last = meta_data.get("is_last", False)
                        defects = meta_data.get("defects", [])

                        # 3. 保存文件
                        pos_dir = self.output_root / f"pos{pos_id}"
                        pos_dir.mkdir(parents=True, exist_ok=True)
                        
                        cv2.imwrite(str(pos_dir / filename), image)

                        if defects:
                            with open(pos_dir / "defects.json", 'w', encoding='utf-8') as f:
                                json.dump(defects, f, indent=2, ensure_ascii=False)

                        # 4. 发送信号给 UI
                        # 只有当收到最后一张图，或者重要的 phase 图时才刷新界面，避免刷新太频繁
                        # 这里逻辑：每收到一张都发信号，UI层决定刷不刷
                        self.data_received.emit(pos_id, is_last)

                finally:
                    conn.close()

        except Exception as e:
            self.log_message.emit(f"[Receiver] 致命错误: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()

    def stop(self):
        self.is_running = False
        self.wait()

# =============================================================================
# 线程 2: 机器人模拟控制线程 (前身是 server.py + 模拟逻辑)
# =============================================================================
class RobotSimulationThread(QThread):
    log_message = pyqtSignal(str)
    task_finished = pyqtSignal()

    def __init__(self, project_folder: str):
        super().__init__()
        self.project_folder = project_folder
        self.host_control = None
        self.is_running = False

    def run(self):
        self.is_running = True
        try:
            self.log_message.emit("[Control] 初始化 HostControl...")
            self.host_control = HostControl(self.project_folder)
            
            # 初始化连接 (这一步会阻塞直到从机 client.py 连接 4096)
            self.log_message.emit("[Control] 等待从机连接控制端口 4096...")
            self.host_control.Socket_init() 
            self.log_message.emit("[Control] 从机已连接！初始化投影...")
            self.host_control.Projected_init()

            # === 模拟自动检测流程 ===
            # 假设我们要检测 3 个点位
            target_points = [1, 2, 3] 

            for p in target_points:
                if not self.is_running: break

                # 1. 模拟机械臂移动
                self.log_message.emit(f"[Robot] >>> 机械臂正在移动到点位 {p} ...")
                time.sleep(2) # 模拟耗时

                # 2. 视觉检测
                self.log_message.emit(f"[Robot] 到位 {p}，触发视觉拍照...")
                # 注意：Take_photo 里面应该包含等待从机拍摄完成的逻辑(wait for switch_pattern)
                # 这样才能保证是同步的
                self.host_control.Take_photo() 
                
                self.log_message.emit(f"[Robot] 点位 {p} 采集指令已发送")
                
                # 等待一会儿，防止连续触发太快
                time.sleep(1) 

            self.log_message.emit("[Control] 所有任务已完成")

        except Exception as e:
            self.log_message.emit(f"[Control] 异常: {e}")
        finally:
            if self.host_control:
                self.host_control.Disconnect()
            self.task_finished.emit()

    def stop(self):
        self.is_running = False
        # 注意：如果是 socket 阻塞，这里可能停不下来，需要更复杂的处理
        # 简单 demo 中我们依靠 is_running 标志位

# =============================================================================
# 主窗口类 (集成 UI 和 线程)
# =============================================================================
class DefectVisualizer(QMainWindow):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = Path(data_root)
        
        # 定义线程引用
        self.receiver_thread = None
        self.robot_thread = None
        
        self.setStyleSheet(DARK_STYLESHEET)
        self.init_ui()
        self.load_pos_list()
        
        # 启动数据接收后台服务 (一直运行)
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
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_title = QLabel("POSITIONS")
        lbl_title.setObjectName("TitleLabel")
        left_layout.addWidget(lbl_title)

        self.pos_list_widget = QListWidget()
        self.pos_list_widget.currentRowChanged.connect(self.on_pos_selected)
        left_layout.addWidget(self.pos_list_widget)
        
        # --- 新增：控制按钮区 ---
        btn_layout = QVBoxLayout()
        self.btn_start = QPushButton("开始自动检测")
        self.btn_start.clicked.connect(self.toggle_robot_task)
        self.btn_start.setMinimumHeight(40)
        
        btn_layout.addWidget(self.btn_start)
        left_layout.addLayout(btn_layout)
        # ---------------------

        lbl_tip = QLabel("Tip: A/D 切图, 滚轮缩放")
        lbl_tip.setStyleSheet("color: #888; font-size: 9pt;")
        left_layout.addWidget(lbl_tip)

        # === 中间面板 (保持不变) ===
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)

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
        
        # 自定义 GraphicsView 类省略了，请确保 ZoomableGraphicsView 类在文件中定义了
        # 这里直接复用之前的 ZoomableGraphicsView
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QBrush(QColor("#1e1e1e")))
        self.view = ZoomableGraphicsView(self.scene)
        
        center_layout.addLayout(tool_layout)
        center_layout.addWidget(self.view)

        # === 右侧面板 (保持不变) ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 统计模块
        stats_group = QGroupBox("STATISTICS")
        stats_layout = QVBoxLayout()
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["类型", "数量"])
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.setFixedHeight(80) 
        self.stats_table.setShowGrid(False) 
        self.stats_table.setStyleSheet("border: none; background-color: transparent;")
        stats_layout.addWidget(self.stats_table)
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group, stretch=1)

        # 列表模块
        list_group = QGroupBox("DEFECT LIST")
        list_layout = QVBoxLayout()
        self.defect_table = QTableWidget()
        self.defect_table.setColumnCount(3)
        self.defect_table.setHorizontalHeaderLabels(["ID", "类型", "Conf"])
        self.defect_table.verticalHeader().setVisible(False)
        self.defect_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.defect_table.cellClicked.connect(self.on_table_row_clicked)
        header = self.defect_table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.defect_table.setColumnWidth(0, 40)
        self.defect_table.setColumnWidth(2, 60)
        list_layout.addWidget(self.defect_table)
        list_group.setLayout(list_layout)
        right_layout.addWidget(list_group, stretch=2)

        # 特写模块
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

    # === 业务逻辑 ===

    def start_receiver_service(self):
        """启动后台接收服务"""
        self.receiver_thread = DataReceiverThread(self.data_root)
        self.receiver_thread.data_received.connect(self.on_new_data_received)
        self.receiver_thread.log_message.connect(lambda msg: print(msg)) # 这里可以改成在界面Status栏显示
        self.receiver_thread.start()

    def toggle_robot_task(self):
        """启动/停止 机械臂检测任务"""
        if self.robot_thread and self.robot_thread.isRunning():
            # 停止任务
            self.robot_thread.stop()
            self.btn_start.setText("正在停止...")
            self.btn_start.setEnabled(False)
        else:
            # 启动任务
            # TODO: 请修改这里的 pattern 路径
            project_folder = r'D:\Project\CarPaintDetection\framework\cfg\patterns\nums10' 
            
            self.robot_thread = RobotSimulationThread(project_folder)
            self.robot_thread.log_message.connect(self.lbl_pos_info.setText) # 把日志显示在顶部信息栏
            self.robot_thread.task_finished.connect(self.on_robot_task_finished)
            self.robot_thread.start()
            
            self.btn_start.setText("停止检测")
            self.btn_start.setStyleSheet("background-color: #d81e06; color: white;") # 变红

    def on_robot_task_finished(self):
        self.btn_start.setText("开始自动检测")
        self.btn_start.setStyleSheet("") # 恢复默认颜色
        self.btn_start.setEnabled(True)
        self.lbl_pos_info.setText("检测任务结束")

    def on_new_data_received(self, pos_id, is_last):
        """当后台收到新数据时触发"""
        # 1. 刷新左侧列表 (检查是否需要新增 item)
        dir_name = f"pos{pos_id}"
        items = self.pos_list_widget.findItems(dir_name, Qt.MatchExactly)
        if not items:
            self.pos_list_widget.addItem(dir_name)
            # 自动滚动到底部
            self.pos_list_widget.scrollToBottom()
        
        # 2. 如果是该点位的最后一张图，且是实时监控模式（这里默认开启），自动选中该行
        if is_last:
            items = self.pos_list_widget.findItems(dir_name, Qt.MatchExactly)
            if items:
                self.pos_list_widget.setCurrentItem(items[0])
                # 触发加载
                self.on_pos_selected(self.pos_list_widget.row(items[0]))

    # --- 以下是之前的 View 逻辑，保持不变 ---
    # 为了节省篇幅，这里复用你之前的逻辑
    # 只要把你之前的 keyPressEvent, showEvent, refresh_view 等等函数粘回来即可
    # 记得把 ZoomableGraphicsView 和 InteractiveDefectBox 类也放进去
    
    # ... (粘贴你之前的 load_pos_list, on_pos_selected, load_defects_data 等方法) ...
    # 确保 refresh_view, draw_defects, update_roi_view 等都在
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            current_row = self.pos_list_widget.currentRow()
            if current_row > 0:
                self.pos_list_widget.setCurrentRow(current_row - 1)
        elif event.key() == Qt.Key_D:
            current_row = self.pos_list_widget.currentRow()
            if current_row < self.pos_list_widget.count() - 1:
                self.pos_list_widget.setCurrentRow(current_row + 1)
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(100, self.fit_image)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_image()

    def fit_image(self):
        if self.scene.itemsBoundingRect().width() > 0:
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def load_pos_list(self):
        self.pos_list_widget.clear()
        if not self.data_root.exists(): return
        dirs = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("pos")]
        dirs.sort(key=lambda x: int(x.name.replace("pos", "")) if x.name.replace("pos", "").isdigit() else 0)
        for d in dirs:
            self.pos_list_widget.addItem(d.name)

    def on_pos_selected(self, index):
        if index < 0: return
        item = self.pos_list_widget.item(index)
        self.current_pos_dir = self.data_root / item.text()
        self.lbl_pos_info.setText(f"CURRENT: {item.text()}")
        
        self.load_defects_data()
        self.update_stats_table()
        self.update_defects_table()
        self.refresh_view()

    def load_defects_data(self):
        self.defects_data = []
        json_path = self.current_pos_dir / "defects.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    raw_data = json.load(f)
                if len(raw_data) > 0 and isinstance(raw_data[0], list):
                     if len(raw_data[0]) > 0 and isinstance(raw_data[0][0], list):
                         self.defects_data = [d for sublist in raw_data for d in sublist]
                     else:
                         self.defects_data = [d for sublist in raw_data for d in sublist]
                else:
                    self.defects_data = raw_data
            except Exception:
                pass

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
            self.fit_image()
        else:
            self.scene.addText(f"Image Not Found: {img_name}", QFont("Arial", 20)).setDefaultTextColor(Qt.red)

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
            if i == index:
                box.set_highlight(True)
            else:
                box.set_highlight(False)
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
            
            label_size = self.roi_label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.roi_label.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    local_output_path = Path(__file__).parent.parent.parent / "output" 
    
    app = QApplication(sys.argv)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    window = DefectVisualizer(data_root=local_output_path)
    window.showMaximized() 
    sys.exit(app.exec_())