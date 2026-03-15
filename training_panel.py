from dataclasses import dataclass  # simple, clean “data container”
from PyQt6.QtCore import Qt, QTimer, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame, QWidget, QLabel, QComboBox, QSlider, QSpinBox,
    QFormLayout, QHBoxLayout, QVBoxLayout, QPushButton, QGridLayout, QSizePolicy
)

@dataclass
class DataSet:
    size_of_training_data: int = 60_000
    selected_dataset: str = "MNIST"
    training: bool = False
    stepRequested: bool = False

class TrainingPanel(QFrame):
    predictRequested = pyqtSignal()
    stepRequested = pyqtSignal()
    previousRequested = pyqtSignal()
    trainingToggled = pyqtSignal(bool)
    datasetChanged = pyqtSignal(str)
    trainSizeChanged = pyqtSignal(int)
    configChanged = pyqtSignal(object)
    def __init__(self):
        super().__init__()

        self.setObjectName("training_panel")

        self.setFrameShape(QFrame.Shape.StyledPanel)

        self.config = DataSet()

        title = QLabel("Training Panel")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        title.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self.predict_btn = QPushButton("Predict")

        self.dataset_selector = QComboBox()
        self.dataset_selector.addItems(["MNIST"])

        self.play_pause_btn = QPushButton("Play/Pause")
        self.play_pause_btn.setCheckable(True)

        self.previous_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")

        self.size_of_training_data_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_of_training_data_slider.setMinimum(100)
        self.size_of_training_data_slider.setMaximum(60_000)
        self.size_of_training_data_slider.setValue(self.config.size_of_training_data)
        self.size_of_training_data_slider.setSingleStep(100)
        self.size_of_training_data_slider.setPageStep(100)

        self.size_of_training_data_slider_title = QLabel("Size of Training Data")
        self.size_of_training_data_slider_label = QLabel("60000")

        # Add functionality

        self.predict_btn.clicked.connect(self.predict)
        self.dataset_selector.currentTextChanged.connect(self.select_data)
        self.play_pause_btn.toggled.connect(self.set_play_pause)
        self.next_btn.clicked.connect(self.step)
        self.size_of_training_data_slider.valueChanged.connect(self.on_train_size_changed)
        self.previous_btn.clicked.connect(self.load_previous)


        root = QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(20)
        root.addWidget(title)

        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(10,10,10,10)
        grid_layout.setSpacing(20)

        # predict
        grid_layout.addWidget(self.predict_btn, 0, 0)

        # dataSet
        grid_layout.addWidget(self.dataset_selector, 0, 1)

        # Training Control Buttons
        training_control_buttons_layout = QHBoxLayout()

        training_control_buttons_layout.addWidget(self.previous_btn,0)
        training_control_buttons_layout.addWidget(self.play_pause_btn, 0)
        training_control_buttons_layout.addWidget(self.next_btn, 0)

        grid_layout.addLayout(training_control_buttons_layout, 1, 0)

        # size of training data slider
        size_of_training_data_layout = QVBoxLayout()
        slider_layout = QHBoxLayout()

        size_of_training_data_layout.setContentsMargins(0,0,0,0)
        size_of_training_data_layout.setSpacing(0)

        size_of_training_data_layout.addWidget(self.size_of_training_data_slider_title, 0)
        size_of_training_data_layout.setSpacing(20)

        slider_layout.addWidget(self.size_of_training_data_slider, 0)
        slider_layout.addWidget(self.size_of_training_data_slider_label, 0)

        size_of_training_data_layout.addLayout(slider_layout, 0)

        grid_layout.addLayout(size_of_training_data_layout, 1, 1)
        root.addLayout(grid_layout)

    def update_ui(self):
        size_of_training_data = self.size_of_training_data_slider.value()
        self.size_of_training_data_slider_label.setText(str(size_of_training_data))

    def update_config_from_ui(self):
        self.config.size_of_training_data = self.size_of_training_data_slider.value()
        self.config.selected_dataset = self.dataset_selector.currentText()

    def select_data(self, text: str):
        self.config.selected_dataset = text
        self.datasetChanged.emit(text)

    def set_play_pause(self, checked:bool):
        self.config.training = checked
        self.trainingToggled.emit(checked)

    def on_train_size_changed(self, value):
        step = 100

        snapped = round(value / step) * step

        if snapped != value:
            self.size_of_training_data_slider.blockSignals(True)
            self.size_of_training_data_slider.setValue(snapped)
            self.size_of_training_data_slider.blockSignals(False)
            value = snapped

        self.update_ui()
        self.update_config_from_ui()
        self.trainSizeChanged.emit(value)

    def on_dataset_changed(self, text):
        self.update_config_from_ui()

    def predict(self):
        self.predictRequested.emit()

    def step(self):
        self.stepRequested.emit()

    def load_previous(self):
        self.previousRequested.emit()

    def emit_config(self):
        self.configChanged.emit(self.config)






