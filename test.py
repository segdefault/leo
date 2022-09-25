from PyQt5.QtCore import Qt, QEvent, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QFormLayout, QLabel, QPushButton, QSlider
import functools
import numpy as np
import keras
import cv2
import sys

import preprocess
import consts


def increase_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l2 = clahe.apply(l)

    lab = cv2.merge((l2, a, b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def toQImage(rgbImage, size=None):
    h, w, ch = rgbImage.shape
    bytesPerLine = ch * w
    convertToQtFormat = QImage(
        rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)

    if size is None:
        return convertToQtFormat
    else:
        return convertToQtFormat.scaled(size[0], size[1], Qt.KeepAspectRatio)


class HCIThread(QThread):
    update = pyqtSignal(QImage, QImage)

    def __init__(self, parent):
        super(HCIThread, self).__init__(parent)

        self.generator = keras.models.load_model(consts.generator_path)
        self.running = False
        self.stopped = False
        self.redraw = False

    def stop(self):
        self.running = False

        while not self.stopped:
            QThread.sleep(1)

    def run(self):
        self.running = True

        while self.running and not self.isInterruptionRequested():
            if self.redraw:
                self.generate()
                self.redraw = False

            QThread.usleep(50)

        self.stopped = True

    def generate(self):
        sketch_input = cv2.resize(self.sketch / 255., consts.image_size)
        image = sketch_input.reshape((1,) + consts.input_shape)
        image = self.generator(image)

        sketch_preview = self.sketch.astype(np.uint8)

        image_preview = image.numpy().reshape(consts.output_shape)
        image_preview = (image_preview - image_preview.min()) / \
            (image_preview.max() - image_preview.min())
        image_preview = (image_preview * 255.).astype(np.uint8)

        qSketch = toQImage(sketch_preview, consts.canvas_size)
        qImage = toQImage(image_preview, consts.canvas_size)

        self.update.emit(qSketch, qImage)


class LeoWindow(QMainWindow):
    def __init__(self, app):
        super(LeoWindow, self).__init__()

        self.brush_radius = 1
        self.brush_color = (0, 0, 0)
        self.initUI()

        self.thread = HCIThread(self)
        app.aboutToQuit.connect(self.thread.stop)
        self.thread.update.connect(self.update)
        self.thread.start()
        self.clear()

    def initUI(self):
        self.canvas = QLabel()
        self.output = QLabel()
        clear_button = QPushButton("CLEAR")
        brush_slider = QSlider(Qt.Horizontal)

        brush_slider.setMinimum(1)
        brush_slider.setMaximum(128)
        brush_slider.setValue(self.brush_radius)
        brush_slider.setTickPosition(QSlider.TicksBothSides)

        canvas_layout = QGridLayout()
        canvas_container = QWidget()
        canvas_container.setLayout(canvas_layout)

        palette_layout = QFormLayout()
        palette_container = QWidget()
        palette_container.setLayout(palette_layout)

        layout = QGridLayout()
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        def add_color(name, color):
            color = (color[2], color[1], color[0]) # BGR to RGB
            label = QLabel(name.upper())
            button = QPushButton()

            button.setStyleSheet('background-color: rgb'+str(color)+';')
            button.clicked.connect(
                functools.partial(
                    self.set_brush_color,
                    color,
                ),
            )

            palette_layout.addRow(label, button)

        for mask_type, value in consts.palette.items():
            if mask_type == 'hair':
                for name, color in value.items():
                    print(color)
                    add_color(f"Hair: {name}", color)
            else:
                add_color(mask_type.capitalize(), value)


        canvas_layout.addWidget(palette_container, 0, 0, 3, 1)
        canvas_layout.addWidget(brush_slider, 0, 1, alignment=Qt.AlignBottom)
        canvas_layout.addWidget(self.canvas, 1, 1, alignment=Qt.AlignCenter)
        canvas_layout.addWidget(clear_button, 2, 1, alignment=Qt.AlignTop)

        layout.addWidget(canvas_container, 0, 0, alignment=Qt.AlignCenter)
        layout.addWidget(self.output, 0, 1, alignment=Qt.AlignCenter)

        container.setStyleSheet("background-color: #EEE;")

        self.canvas.installEventFilter(self)
        clear_button.clicked.connect(self.clear)
        brush_slider.valueChanged.connect(
            lambda: self.set_brush_radius(brush_slider.value()))

    @pyqtSlot(QImage, QImage)
    def update(self, sketch, image):
        self.canvas.setPixmap(QPixmap.fromImage(sketch))
        self.output.setPixmap(QPixmap.fromImage(image))

    def clear(self):
        self.thread.sketch = np.zeros(consts.canvas_size + (3,))
        self.thread.redraw = True

    def set_brush_radius(self, radius):
        self.brush_radius = radius

    def set_brush_color(self, color):
        print(color)
        self.brush_color = color

    def eventFilter(self, source, event):
        if (event.type() in (QEvent.MouseButtonPress, QEvent.MouseMove)
                and source is self.canvas):
            self.paint(event)

        return QWidget.eventFilter(self, source, event)

    def paint(self, event):
        x = max(0, min(int(event.localPos().x()), consts.canvas_size[0] - 1))
        y = max(0, min(int(event.localPos().y()), consts.canvas_size[1] - 1))
        brush = event.buttons() == Qt.LeftButton
        value = self.thread.sketch[y][x]

        if brush:
            cv2.circle(self.thread.sketch, (x, y), self.brush_radius,
                       self.brush_color, thickness=-1)
        else:
            cv2.circle(self.thread.sketch, (x, y),
                       self.brush_radius, (0, 0, 0), thickness=-1)

        self.thread.redraw = True


def main():
    app = QApplication([])

    window = LeoWindow(app)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
