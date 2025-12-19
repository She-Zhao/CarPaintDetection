import sys
import json
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QListWidget, QLabel, QGraphicsView, 
                             QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
                             QFrame, QSplitter, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QAbstractItemView, QGroupBox, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QBrush, QPainter, QFont

# --- 全局样式表 (Dark Theme) ---
# 修改点：增加了 ComboBox 的详细配色，确保文字清晰
DARK_STYLESHEET = """
QMainWindow {
    background-color: #2b2b2b;
}
QWidget {
    color: #e0e0e0;
    font-family: "Segoe UI", "Microsoft YaHei";
    font-size: 10pt;
}
QGroupBox {
    border: 1px solid #444;
    border-radius: 5px;
    margin-top: 10px;
    font-weight: bold;
    background-color: #333;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: #aaa;
}
QListWidget, QTableWidget {
    background-color: #3c3f41;
    border: 1px solid #555;
    border-radius: 4px;
    gridline-color: #444;
}
QListWidget::item:selected, QTableWidget::item:selected {
    background-color: #0078d7;
    color: white;
}
QListWidget::item:hover, QTableWidget::item:hover {
    background-color: #4c5052;
}
QHeaderView::section {
    background-color: #444;
    color: #ccc;
    padding: 4px;
    border: none;
    border-bottom: 1px solid #555;
}
/* --- 修复点：优化下拉框样式 --- */
QComboBox {
    background-color: #3c3f41;
    color: #ffffff;  /* 强制白字 */
    border: 1px solid #555;
    padding: 5px;
    border-radius: 4px;
}
QComboBox:hover {
    border: 1px solid #0078d7;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
/* 下拉弹窗的样式 */
QComboBox QAbstractItemView {
    background-color: #3c3f41;
    color: #ffffff;
    selection-background-color: #0078d7;
    border: 1px solid #555;
}
/* ------------------------- */
QScrollBar:vertical {
    background: #2b2b2b;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #555;
    border-radius: 5px;
}
QLabel#TitleLabel {
    font-size: 12pt;
    font-weight: bold;
    color: #00aaff;
}
"""

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

# --- 自定义视图：支持鼠标缩放 ---
class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag) 
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse) 
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background-color: #1e1e1e; border: none;")

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

# --- 自定义图形项 ---
class InteractiveDefectBox(QGraphicsRectItem):
    def __init__(self, x, y, w, h, defect_id, on_click_callback):
        super().__init__(x, y, w, h)
        self.defect_id = defect_id
        self.on_click_callback = on_click_callback
        self.setAcceptHoverEvents(True)
        self.default_pen = QPen(QColor(255, 50, 50), 3) 
        self.highlight_pen = QPen(QColor(0, 255, 255), 5) 
        self.setPen(self.default_pen)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.on_click_callback(self.defect_id)
            event.accept()

    def set_highlight(self, active: bool):
        if active:
            self.setPen(self.highlight_pen)
            self.setZValue(10)
        else:
            self.setPen(self.default_pen)
            self.setZValue(0)

# --- 主窗口 ---
class DefectVisualizer(QMainWindow):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = Path(data_root)
        self.current_pos_dir = None
        self.current_img_np = None
        self.defects_data = []
        self.box_items = []
        
        self.setStyleSheet(DARK_STYLESHEET)
        
        self.init_ui()
        self.load_pos_list()

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
        
        lbl_tip = QLabel("Tip: 使用 A / D 键切换点位\n鼠标滚轮缩放图像")
        lbl_tip.setStyleSheet("color: #888; font-size: 9pt;")
        left_layout.addWidget(lbl_tip)

        # === 中间面板 ===
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
        
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QBrush(QColor("#1e1e1e")))
        self.view = ZoomableGraphicsView(self.scene)
        
        center_layout.addLayout(tool_layout)
        center_layout.addWidget(self.view)

        # === 右侧面板 ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 1. 统计模块
        stats_group = QGroupBox("STATISTICS (统计)")
        stats_layout = QVBoxLayout()
        stats_layout.setContentsMargins(5, 10, 5, 5)
        
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["缺陷类型", "数量"])
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.setFixedHeight(80) 
        self.stats_table.setShowGrid(False) 
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.stats_table.setStyleSheet("border: none; background-color: transparent;")
        
        stats_layout.addWidget(self.stats_table)
        stats_group.setLayout(stats_layout)
        
        # 修改点：设置 stretch=0，让它只占最小空间
        right_layout.addWidget(stats_group, stretch=1)

        # 2. 缺陷列表模块
        list_group = QGroupBox("DEFECT LIST (列表)")
        list_layout = QVBoxLayout()
        list_layout.setContentsMargins(5, 10, 5, 5)
        
        self.defect_table = QTableWidget()
        self.defect_table.setColumnCount(3)
        self.defect_table.setHorizontalHeaderLabels(["ID", "类型", "Conf"])
        self.defect_table.verticalHeader().setVisible(False)
        self.defect_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.defect_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.defect_table.cellClicked.connect(self.on_table_row_clicked)
        
        header = self.defect_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        self.defect_table.setColumnWidth(0, 40)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self.defect_table.setColumnWidth(2, 60)
        
        list_layout.addWidget(self.defect_table)
        list_group.setLayout(list_layout)
        
        # 修改点：设置 stretch=2，占比适中
        right_layout.addWidget(list_group, stretch=2)

        # 3. 局部特写模块
        roi_group = QGroupBox("INSPECTION (特写)")
        roi_layout = QVBoxLayout()
        roi_layout.setContentsMargins(5, 10, 5, 5)
        
        self.roi_label = QLabel("No Selection")
        self.roi_label.setAlignment(Qt.AlignCenter)
        self.roi_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # 允许自由扩展
        self.roi_label.setStyleSheet("background-color: #000; border: 1px solid #555;")
        self.roi_label.setScaledContents(False) 
        
        roi_layout.addWidget(self.roi_label)
        roi_group.setLayout(roi_layout)
        
        # 修改点：设置 stretch=3，占比最大，保证特写区域足够大
        right_layout.addWidget(roi_group, stretch=3)

        # === 布局分割器 ===
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 2)
        
        main_layout.addWidget(splitter)

    # --- 逻辑部分保持不变 ---
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