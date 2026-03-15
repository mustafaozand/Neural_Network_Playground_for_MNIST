# from __future__ import annotations

import math
import random

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union


from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QBrush, QGuiApplication, QAction, QPixmap, QColor
from PyQt6.QtWidgets import QFrame, QToolTip, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QApplication

import sys
import torch


class OutputPanel(QFrame):
    def __init__(self):
        super().__init__()

        self.setMouseTracking(True)

        self.predicted_values = torch.tensor([[0,0,0,0,0,0,0,0,0,0]])

        self.margin = 24
        self.node_rad = 24
        self.node_gap = 100

        self.node_hitboxes = {}

        self.last_hover_key = None

        self.node_pos = []
        self.node_classname = ["0", "1", "2", "3","4","5","6","7","8","9"]
        self.gray_scale_values = []


    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)

        # according to the documentation without antialiasing Qt draws shapes using hard pixel edges this will cause jagged circles aka like rough looking nodes
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        self.draw_nodes(painter)

    def draw_nodes(self, painter):
        w = self.width()
        h = self.height()

        self.node_pos.clear()
        self.node_hitboxes.clear()

        left = self.margin
        top = self.margin
        right = w - self.margin
        bottom = h - self.margin

        # calculating y and x pos of every node
        # calculating positions for every output node (evenly spaced so they stay on the screen)
        total_number_of_output_nodes = int(self.predicted_values.shape[1])

        node_pen = QPen(Qt.GlobalColor.black)
        node_pen.setWidth(1)

        for node in range(total_number_of_output_nodes):
            # if node == 0:
            top_y = top + self.node_rad
            bottom_y = bottom - self.node_rad
            usable_h = max(1.0, float(bottom_y - top_y))

            y_val = (top_y + usable_h * (node + 2) / (total_number_of_output_nodes + 1))
            x_val = (left + right) // 2
            pos =  QPointF(x_val, y_val)
            self.node_pos.append(pos)

        gray_scale = self.compute_grayscale_values()

        for node in range(len(self.node_pos)):

            if gray_scale:
                value = gray_scale[node]
                value = int(round(float(value)))
                value = max(0, min(255, value))
                color = QColor(value, value, value)
                node_brush = QBrush(color)

            painter.setPen(node_pen)
            painter.setBrush(node_brush)

            rect = QRectF(self.node_pos[node].x() - self.node_rad,
                          self.node_pos[node].y() - self.node_rad,
                          2*(self.node_rad), 2*(self.node_rad))
            painter.drawEllipse(rect)

            self.node_hitboxes[node] = rect

    def mouseMoveEvent(self, event):
        pos = event.position()

        for idx, rect in self.node_hitboxes.items():
            if rect.contains(pos):
                prob = float(self.predicted_values[0][idx])
                QToolTip.showText(self.mapToGlobal(pos.toPoint()), f"Class {self.node_classname[idx]}: {prob:.4f}")
                return

            QToolTip.hideText()

    def compute_grayscale_values(self):
        if not isinstance(self.predicted_values, torch.Tensor):
            return []

        preds = self.predicted_values.detach().cpu().float()

        # if shape is (1, 10) then flatten to (10,)
        if preds.dim() == 2:
            preds = preds[0]

        # if empty
        if preds.numel() == 0:
            return []

        max_val = float(preds.max().item())
        # if all the vallues are zero
        if max_val == 0.0:
            return [0.0 for _ in range(preds.numel())]

        return [(float(v.item()) / max_val) * 255.0 for v in preds]

        # This version for the mouseMoveEvent failed
        # pos = event.pos()
        # class_index = self.node_pos.index(pos.toPointF())
        # hit = ("node", class_index, self.predicted_values[class_index], self.node_classname[class_index])
        #
        # if hit is None:
        #     if self.last_hover_key  is not None:
        #         QToolTip.hideText()
        #         self.last_hover_key = None
        #
        #     return
        #
        # class_name, probabilities = self.node_classname, self.predicted_values
        # hover_key = (class_name, probabilities)
        #
        # if hover_key != self.last_hover_key:
        #     QToolTip.showText(self.mapToGlobal(pos.toPoint()), str(hit[3]), self)
        #     self.last_hover_key = hover_key

    def leaveEvent(self, event):
        QToolTip.hideText()
        self.last_hover_key = None
        super().leaveEvent(event)


    def set_predicted_values(self, predicted_values):
        self.predicted_values = predicted_values



# Demo Application
if __name__ == "__main__":
    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)

    window = QMainWindow()
    window.setWindowTitle("OutputPanel Demo")

    panel = OutputPanel()
    panel.setMinimumSize(300, 600)

    # Simple demo: update predictions every second so you can see the grayscale change.
    def update_random_predictions():
        # Generate a random probability-like vector and normalise it so it sums to 1.
        x = torch.rand(1, 10)
        x = x / x.sum(dim=1, keepdim=True)
        panel.set_predicted_values(x)
        panel.update()  # trigger repaint

    timer = QTimer()
    timer.timeout.connect(update_random_predictions)
    timer.start(1000)

    # Set initial values immediately
    update_random_predictions()

    window.setCentralWidget(panel)
    window.resize(360, 640)
    window.show()

    sys.exit(app.exec())




