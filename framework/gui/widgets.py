# framework/gui/widgets.py

from PyQt5.QtWidgets import QGraphicsView, QGraphicsRectItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QColor

class ZoomableGraphicsView(QGraphicsView):
    """支持鼠标滚轮缩放的视图"""
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

class InteractiveDefectBox(QGraphicsRectItem):
    """可点击的缺陷矩形框"""
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