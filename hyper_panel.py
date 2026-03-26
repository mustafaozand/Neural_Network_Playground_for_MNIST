from dataclasses import dataclass  # simple, clean “data container”
from PyQt6.QtCore import Qt, pyqtSignal, QTimer  # signals + a timer for debouncing
from PyQt6.QtWidgets import (
    QFrame, QWidget, QLabel, QComboBox, QSlider, QSpinBox,
    QFormLayout, QHBoxLayout, QVBoxLayout
)


@dataclass(slots=False)
class HyperParams:
    lr: float =  0.01
    batch_size: int = 32
    activation: str = "ReLU"
    optimiser: str = "SGD"


# self.debounce = QTimer(self)
# self.debounce.setSingleShot(True)
# self.debounce.timeout.connect(self.emit_config)

class HyperParameterPanel(QFrame):
    configChanged = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.setObjectName("hyper_panel")

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.config = HyperParams()
        title = QLabel("Hyperparameters")

        title.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)

        # learning rate slider
        self.lr_slider = QSlider(Qt.Orientation.Horizontal)
        self.lr_slider.setMinimum(10)
        self.lr_slider.setMaximum(60)
        self.lr_slider.setValue(0)
        self.lr_value = QLabel("")

        #batch size spinbox
        self.batch_size = QSpinBox()
        self.batch_size.setMinimum(1)
        self.batch_size.setMaximum(64)
        self.batch_size.setValue(self.config.batch_size)

        #the activation function combobox
        self.activation_func = QComboBox()
        self.activation_func.addItems(["ReLU", "Sigmoid","Tanh"])

        #the optimisers combobox
        self.optimiser = QComboBox()
        self.optimiser.addItems(["SGD","Adam"])

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        root.addWidget(title)

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)

        lr_row = QWidget()
        lr_row_layout = QHBoxLayout(lr_row)
        lr_row_layout.setContentsMargins(0, 0, 0, 0)
        lr_row_layout.setSpacing(12)
        lr_row_layout.addWidget(self.lr_slider,1)
        lr_row_layout.addWidget(self.lr_value)
        form.addRow("Learning Rate", lr_row)

        form.addRow("Activation Function", self.activation_func)
        form.addRow("Optimiser", self.optimiser)
        form.addRow("Batch Size", self.batch_size)

        root.addLayout(form)
        root.addStretch()

        self.lr_slider.valueChanged.connect(self.on_lr_changed)

        self.batch_size.valueChanged.connect(self.on_any_value_changed)
        self.activation_func.currentTextChanged.connect(self.on_any_value_changed)
        self.optimiser.currentTextChanged.connect(self.on_any_value_changed)

    # sliders in pyqt only accept integers this function converts the integer value of the slider into float
    def convert_lr_slider_int_to_corresponding_float(self,s):
        return 10 ** (-6 + (s/10.0))


    def update_ui(self):

        lr = self.convert_lr_slider_int_to_corresponding_float(self.lr_slider.value())
        self.lr_value.setText(f"{lr: .5f}")

    # whenever the user changes a value inside the hyperparameter panel this method gets called
    def update_config_from_ui(self):
        self.config.lr = self.convert_lr_slider_int_to_corresponding_float(self.lr_slider.value())
        self.config.batch_size = int(self.batch_size.value())
        self.config.activation = self.activation_func.currentText()
        self.config.optimiser = self.optimiser.currentText()

    # when the learning rate is changed this method is called
    def on_lr_changed(self, value:int):
        self.update_ui()
        self.update_config_from_ui()
        self.configChanged.emit(self.config)

    # this method is called to change the hyperparameter values (config) and update UI components
    # for this only the label that shows the user the learning rate is updated because the other components are handled by pyqt
    def on_any_value_changed(self):
        self.update_ui()
        self.update_config_from_ui()
        self.configChanged.emit(self.config)

    # getter method for config
    def get_config(self):
        self.update_config_from_ui()
        return self.config



















