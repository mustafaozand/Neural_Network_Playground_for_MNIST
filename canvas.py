import sys

import torch
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPen, QGuiApplication
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QMainWindow, QSlider, QLabel, QHBoxLayout, QFrame,
                             QGridLayout)


class CanvasPanel(QFrame):
    def __init__(self, canvas_size):
        super().__init__()

        self.canvas_size = canvas_size
        self.setFixedSize(self.canvas_size[0], self.canvas_size[1])

        self.image = QImage(self.canvas_size[0], self.canvas_size[1], QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)

        self.drawing = False
        self.brush_size = 10
        self.brush_color = Qt.GlobalColor.black

        self.last_pos = QPoint()

    def clear(self):
        self.image.fill(Qt.GlobalColor.white)
        self.update()

    def set_pen_size(self, size: int):
        self.brush_size = int(size)

    def set_eraser(self, on:bool):
        self.brush_color = Qt.GlobalColor.white if on else Qt.GlobalColor.black

    def mousePressEvent(self,event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            pen = QPen(self.brush_color)
            pen.setWidth(self.brush_size)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)

            painter.drawLine(self.last_pos, event.pos())
            self.last_pos = event.pos()
            self.update()  # trigger repaint

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(0, 0, self.image)

    def resizeEvent(self, event):
        #resize the image to match the new display size when resized

        new_size = event.size()
        if new_size.width() <= 0 or new_size.height() <= 0:
            return

        if new_size != self.image.size():
            new_image = QImage(new_size, QImage.Format.Format_RGB32)
            new_image.fill(Qt.GlobalColor.white)

            p = QPainter(new_image)
            p.drawImage(0, 0, self.image)
            p.end()

            self.image = new_image

        super().resizeEvent(event)

    def convert_img_to_tensor(self):
        # image from the canvas is converted to grayscale
        img = self.image.convertToFormat(QImage.Format.Format_Grayscale8)
        # scaled so that it is 28 by 28 which is the required size for the input layer
        img = img.scaled(28,28, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)

        # the pointer which points to the img in memory
        pointer = img.bits()
        # size of the pointer is set so that only the image is pointed to
        pointer.setsize(img.width() * img.height())
        # convert the img in memory to python readable objects
        data = bytes(pointer)

        # img is converted into pytorch tensors
        pixels = torch.tensor(list(data), dtype=torch.float32)
        # normalised, but this time every pixel value has a value in
        # the range 0 to 1 rather than 0.01 to 0.99
        pixels = (255.0 - pixels) / 255.0
        # the tensor is reshaped so that its (1,784)
        pixels = pixels.view(1, 28*28)

        return pixels

    # TESTING how well the image from the canvas is converted, as there is an issue with the model having a high accuracy which
    # can be seen when accuracy is plotted after each epoch yet, when testing the model, it doesn't work properly and gives incorrect results

    def get_processed_img(self):
        img = self.image.convertToFormat(QImage.Format.Format_Grayscale8)
        img = img.scaled(28,28, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)

        pointer = img.bits()
        pointer.setsize(img.width() * img.height())
        data = bytes(pointer)

        inv = bytes((255 - b)for b in data)

        q = QImage(inv, 28,28,28,QImage.Format.Format_Grayscale8)

        return q.copy()



class CanvasControlPanel(QWidget):
    clear_requested = pyqtSignal()
    pen_size_changed = pyqtSignal(int)
    eraser_toggled = pyqtSignal(bool)
    def __init__(self):
        super().__init__()

        self.clear_btn = QPushButton("Clear")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(5)
        self.slider.setMaximum(30)
        self.slider.setValue(10)

        self.slider_lbl_val = QLabel("10")

        self.eraser_btn = QPushButton("Eraser")
        self.eraser_btn.setCheckable(True)

        row = QHBoxLayout(self)
        row.setContentsMargins(0,0,0,0)
        row.setSpacing(10)

        row.addWidget(self.clear_btn)
        row.addWidget(self.eraser_btn)

        row.addWidget(QLabel("Size"))
        row.addWidget(self.slider,1)
        row.addWidget(self.slider_lbl_val)

        self.clear_btn.clicked.connect(self.clear_requested.emit)
        self.slider.valueChanged.connect(self.on_slider)
        self.eraser_btn.toggled.connect(self.eraser_toggled.emit)

    def on_slider(self, value):
        self.slider_lbl_val.setText(str(value))
        self.pen_size_changed.emit(value)



class CanvasApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Canvas")

        screen = QGuiApplication.primaryScreen()

        if screen is not None:
            self.setGeometry(screen.availableGeometry())

        central = QWidget()
        root = QGridLayout(central)
        root.setContentsMargins(20,20,20,20)
        root.setSpacing(10)

        self.canvas_size = (self.width() // 2, self.height() //2)

        self.canvas_panel = CanvasPanel(self.canvas_size)
        self.canvas_control = CanvasControlPanel()

        self.canvas_control.clear_requested.connect(self.canvas_panel.clear)
        self.canvas_control.pen_size_changed.connect(self.canvas_panel.set_pen_size)
        self.canvas_control.eraser_toggled.connect(self.canvas_panel.set_eraser)

        root.addWidget(self.canvas_panel, 0,0)
        root.addWidget(self.canvas_control, 0,1)


        self.setCentralWidget(central)


def main():
    app = QApplication(sys.argv)
    window = CanvasApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


