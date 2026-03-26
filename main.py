
import os
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QButtonGroup,
                             QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QStackedWidget, QDialog, QTextEdit)
from PyQt6.QtGui import QFont, QPixmap, QGuiApplication
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from canvas import CanvasPanel, CanvasControlPanel
from network_view import NetworkView
from hyper_panel import HyperParameterPanel
from training_panel import TrainingPanel
from output_panel import OutputPanel
from training import Training, SNAPSHOT
from model_performance_visualisation import Analytics
import torch
from device import DEVICE

# the TrainingWorker class is used to run training inside a background thread so that my UI doesn't freeze during training
class TrainingWorker(QObject):
    snapshotReady = pyqtSignal(object) # will emit the SNAPSHOT dictionary after each epoch
    finished = pyqtSignal()

    def __init__(self, trainer: Training):
        super().__init__()
        self.trainer = trainer
        self.running = False

    def run_loop(self):
        self.running = True
        while self.running:
            # run one epoch
            self.trainer.training()

            # tell the UI “new values are ready”
            self.snapshotReady.emit(SNAPSHOT)
        self.finished.emit()

    def stop(self):
        self.running = False

# MainWindow is where objects from each panel are instantiated and also arranged correctly by following my design for
# the Main Window which is in the Design section of my NEA
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classification")
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.availableGeometry())


        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = Sidebar()
        root.addWidget(self.sidebar)

        self.main_page = QStackedWidget()
        self.main_page.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.main_page)


        self.setCentralWidget(central)

        # Stylesheet defines the style for each component of my UI
        self.setStyleSheet("""

        QMainWindow {
            background-color: rgb(75,75,75);
        }

        /* panels */

        #canvas_panel,
        #training_panel,
        #network_view,
        #hyper_panel,
        #output_panel {
            background-color: rgb(95,95,95);
            border-radius: 14px;
        }

        /* sidebar */

        QWidget#Sidebar {
            background-color: #1f2430;
            border-radius: 18px;
        }

        #Sidebar QLabel {
            color: white;
            font-weight: 700;
            font-size: 16px;
        }

        #Sidebar QPushButton {
            font-size: 15px;
            text-align: left;
            padding: 10px 14px;
            border-radius: 12px;
            color: #c7c9d3;
            background: transparent;
            border: none;
        }

        #Sidebar QPushButton:hover {
            background: rgba(255,255,255,0.06);
            color: white;
        }

        #Sidebar QPushButton:checked {
            background-color: #7c5cff;
            color: white;
        }

        /*  buttons */

        QPushButton {
            background-color: rgb(110,110,110);
            color: white;
            border-radius: 8px;
            padding: 6px 10px;
        }
        
        QPushButton:checked {
            background-color: #7c5cff;
            color: white;
        }

        QPushButton:hover {
            background-color: rgb(125,125,125);
        }

        /* labels */

        QLabel {
            color: white;
        }

        /* Sliders */

        QSlider::groove:horizontal {
            height: 6px;
            background: rgb(65,65,65);
            border-radius: 3px;
        }

        QSlider::handle:horizontal {
            background: #7c5cff;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }

        """)

        # these are part of the final iteration
        self.playground_page = QWidget()
        self.analytics_page = QWidget()

        # Information Page
        self.information_page = InformationPage()

        self.main_page.addWidget(self.playground_page)
        self.main_page.addWidget(self.analytics_page)
        self.main_page.addWidget(self.information_page)

        grid_layout = QGridLayout(self.playground_page)
        grid_layout.setContentsMargins(24, 24, 24, 24)
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(20)

        self.canvas_panel = CanvasPanel([400,400])
        canvas_panel_controls = CanvasControlPanel()

        # the dialog is used to display the preprocessed grayscale version of the
        # users input drawn into the canvas
        self.preview_dialog = QDialog(self)
        self.preview_dialog.setWindowTitle("Preview")
        preview_layout = QVBoxLayout(self.preview_dialog)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.preview_label)
        self.preview_dialog.resize(320, 360)

        canvas_panel_controls.clear_requested.connect(self.canvas_panel.clear)
        canvas_panel_controls.pen_size_changed.connect(self.canvas_panel.set_pen_size)
        canvas_panel_controls.eraser_toggled.connect(self.canvas_panel.set_eraser)

        # the training class is instantiated here
        self.trainer = Training()
        self.trainer.set_snapshot()

        # Each of my panels are instantiated here
        training_panel = TrainingPanel()
        network_view = NetworkView()
        hyper_panel = HyperParameterPanel()
        output_panel = OutputPanel()

        # grid_layout.addWidget(canvas_panel, 0, 0)
        # grid_layout.addWidget(training_panel,1,0)

        self.training_panel = training_panel
        self.network_view = network_view
        self.hyper_panel = hyper_panel
        self.output_panel = output_panel

        self.analytics = Analytics(trainer=self.trainer)

        self.analytics_layout = QVBoxLayout(self.analytics_page)
        self.analytics_layout.setContentsMargins(24,24,24,24)
        self.analytics_layout.addWidget(self.analytics)

        #stores a list of older epochs from previous training cycles
        self.snapshot_history = []
        self.history_index = -1

        self.training_panel.previousRequested.connect(self.on_previous_requested)

        # Before threading the UI and the backend was controlled by the train_timer,
        # so when the timer was running because of training, the user couldn't interact with the UI

        # self.train_timer = QTimer(self)
        # self.train_timer.setInterval(0)
        #
        # self.train_timer.timeout.connect(self.train_one_epoch_and_refresh)
        # self.training_panel.trainingToggled.connect(self.on_training_toggled)
        # self.training_panel.stepRequested.connect(self.train_one_epoch_and_refresh)

        # making use of threading
        self.train_thread = QThread(self)
        self.train_worker = TrainingWorker(self.trainer) # change to fit new architecture

        # move the worker object into the thread
        self.train_worker.moveToThread(self.train_thread)

        # when the thread starts, run the worker loop
        self.train_thread.started.connect(self.train_worker.run_loop)

        # when worker produces a snapshot, update the view (runs safely on UI thread)
        self.train_worker.snapshotReady.connect(self.on_snapshot_ready)
        self.train_worker.finished.connect(self.train_thread.quit)

        # connect buttons
        self.training_panel.trainingToggled.connect(self.on_training_toggled)
        self.training_panel.stepRequested.connect(self.on_next_requested)
        # Hyperparameter changes (UI -> backend)
        self.hyper_panel.configChanged.connect(self.on_hyperparams_changed)
        self.training_panel.trainSizeChanged.connect(self.on_training_size_changed)

        self.training_panel.predictRequested.connect(self.on_predict_requested)


        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0,0,0,0)
        left_layout.setSpacing(20)

        left_layout.addWidget(self.canvas_panel, 1)
        left_layout.addWidget(canvas_panel_controls, 0)
        left_layout.addWidget(training_panel, 1)


        grid_layout.addWidget(left_col, 0,0 ,2, 1)

        grid_layout.addWidget(network_view, 1,1)
        grid_layout.addWidget(hyper_panel,0,1,1,2)
        grid_layout.addWidget(output_panel, 1,2)

        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 2)
        grid_layout.setColumnStretch(2,1)

        grid_layout.setRowStretch(0,1)
        grid_layout.setRowStretch(1,3)

        # the object names are used to refer to each of these objects inside my stylesheet, for styling
        self.canvas_panel.setObjectName("canvas_panel")
        training_panel.setObjectName("training_panel")
        network_view.setObjectName("network_view")
        hyper_panel.setObjectName("hyper_panel")
        output_panel.setObjectName("output_panel")

        self.sidebar.group.idClicked.connect(self.main_page.setCurrentIndex)

    # this method detaches my tensors from autograd (training pipeline, it performs the forward, and backward operations. Before I was handling everything but now its handled by torch)
    # then it moves the tensors to the CPU because they could be inside the GPU, and clones them, so that I can use these tensors safely in my code.
    def clone_tensor(self, t):
        if t is None:
            return None
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().clone()
        return t

    def create_snapshot_copy(self, snapshot):
        # create an independent copy of SNAPSHOT for history, so that the previous values of the weights, biases can be stored
        return {
            "weights": [self.clone_tensor(x) for x in snapshot.get("weights", [])],
            "biases": [self.clone_tensor(x) for x in snapshot.get("biases", [])],
            "outputs": [self.clone_tensor(x) for x in snapshot.get("outputs", [])],
            "activated_outputs": [self.clone_tensor(x) for x in snapshot.get("activated_outputs", [])],
            "inputs": [self.clone_tensor(x) for x in snapshot.get("inputs", [])],
        }

    def on_predict_requested(self):
        if self.train_thread.isRunning():
            self.train_worker.stop()
            self.train_thread.quit()
            self.train_thread.wait()
            self.training_panel.play_pause_btn.blockSignals(True)
            self.training_panel.play_pause_btn.setChecked(False)
            self.training_panel.play_pause_btn.blockSignals(False)

        x = self.canvas_panel.convert_img_to_tensor().to(DEVICE)  # shape (1,784) as 28x28= 784

        # this is to show what the user drawn image will look like to the neural network after normalisation
        processed = self.canvas_panel.get_processed_img() # call the method inside the canvas class that will process the img
        user_img = QPixmap.fromImage(processed)
        user_img = user_img.scaled(280,280,Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation)

        # the preview label which is a QLabel will contain the user image
        self.preview_label.setPixmap(user_img)
        self.preview_dialog.show()
        self.preview_dialog.raise_()
        self.preview_dialog.activateWindow()

        # inference mode, this is a feature of pytorch, it prevents gradient tracking,
        # so it basically reduces the use of computational power like GPU memory.
        self.trainer.model.eval()
        with torch.no_grad():
            logits = self.trainer.model(x)  # forward() is called internally
            softmax_values = torch.softmax(logits, dim=1)
            pred_class = int(logits.argmax(dim=1).item())

        self.output_panel.set_predicted_values(softmax_values)

        print(f"Logits:{logits}")
        print(f"Softmax Values: {softmax_values}")
        print(f"Predicted class:{pred_class}")

    def on_training_toggled(self, running: bool):
        if running:
            # start background loop
            if not self.train_thread.isRunning():
                self.train_thread.start()
        else:
            # ask worker to stop looping (stops after current epoch)
            self.train_worker.stop()

    def train_one_epoch_and_refresh(self):
        self.trainer.training()
        self.on_snapshot_ready(SNAPSHOT)


    def on_snapshot_ready(self, snap: dict):
        #called after each epoch from the training thread.
        stored = self.create_snapshot_copy(snap)

        # If the user were to go back in history and then trains again, the previously trained models
        # that come after the current index that the user is viewing will be sliced out of the snapshot_history list
        if self.history_index < len(self.snapshot_history) - 1:
            self.snapshot_history = self.snapshot_history[: self.history_index + 1]
        # snapshot appended to the snapshot_history list
        self.snapshot_history.append(stored)
        self.history_index = len(self.snapshot_history) - 1
        # a new snapshot of the network is displyed
        self.network_view.set_snapshot(stored)
        self.analytics.on_epoch_trained()

    def on_previous_requested(self):
        # load the previous epoch's snapshot
        if self.history_index <= 0:
            return
        self.history_index -= 1
        self.network_view.set_snapshot(self.snapshot_history[self.history_index])

    def on_next_requested(self):
        # if possible, move forward in history, otherwise train one epoch.
        # if we have a newer snapshot stored (user previously pressed previous), just navigate forward.
        if self.history_index < len(self.snapshot_history) - 1:
            self.history_index += 1
            self.network_view.set_snapshot(self.snapshot_history[self.history_index])
            return

        # otherwise we're already at the newest snapshot.
        # don't step train while the background loop is running.
        if self.train_thread.isRunning():
            return

        self.train_one_epoch_and_refresh()

    def on_hyperparams_changed(self, config):
        #called whenever the HyperParameterPanel emits configChanged.
        #stops training if training is running and rebuilds the model/dataloaders/optimiser.

        # If training is currently running in the background thread, stop it first.
        if self.train_thread.isRunning():
            self.train_worker.stop()
            self.train_thread.quit()
            self.train_thread.wait()

            # Make sure the Play/Pause button reflects that training is stopped.
            self.training_panel.play_pause_btn.blockSignals(True)
            self.training_panel.play_pause_btn.setChecked(False)
            self.training_panel.play_pause_btn.blockSignals(False)

        # rebuild the backend training objects using the new hyperparameters
        self.trainer.apply_hyperparams(config) #change to fit new architecture

        # clear history because old snapshots no longer match the new model/settings
        self.snapshot_history.clear()
        self.history_index = -1

        # clear the view until the next epoch produces a new snapshot
        self.trainer.clear_snapshot() #change to fit new architecture


    def on_training_size_changed(self,size:int):
        # if the training thread is running than it is terminated
        if self.train_thread.isRunning():
            self.train_worker.stop()
            self.train_thread.quit()
            self.train_thread.wait()

            # makes sure that the play/pause button reflects that training is stopped.
            self.training_panel.play_pause_btn.blockSignals(True)
            self.training_panel.play_pause_btn.setChecked(False)
            self.training_panel.play_pause_btn.blockSignals(False)

        self.trainer.apply_new_training_size(size)


class Sidebar(QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(180)
        self.setObjectName("Sidebar")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16,16, 16, 16)
        layout.setSpacing(20)

        title = QLabel("Neural Playground")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(title)


        self.group = QButtonGroup(self)
        self.group.setExclusive(True)

        self.btn_play = self.nav_btn("Playground")
        self.btn_analytics = self.nav_btn("Analytics")
        self.btn_instructions = self.nav_btn("Instructions")
        # self.btn_data = self.nav_btn("Dataset")
        # self.btn_learn = self.nav_btn("Learn")

        for i, b in enumerate([self.btn_play, self.btn_analytics, self.btn_instructions]):
            self.group.addButton(b, i)
            layout.addWidget(b)

        layout.addStretch()

        self.btn_play.setChecked(True)

    def nav_btn(self, text: str):
        b = QPushButton(text)
        b.setCheckable(True)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        return b


#Information Page contains useful information about my application
class InformationPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Information Page")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        layout.addWidget(title)

        text = QTextEdit()
        text.setReadOnly(True)

        # Content inside the information page is written in HTML, because it is structured
        text.setHtml("""
        <h1>Neural Playground</h1>

        <p>
        This application allows the user to experiment with a fully connected neural network trained on the MNIST dataset.
        The goal is to provide both a visual and interactive understanding of how neural networks learn.
        </p>

        <h2>MNIST Dataset</h2>
        <p>
        The MNIST dataset consists of 28x28 grayscale images of handwritten digits ranging from 0 to 9.
        Each image is flattened into a 784-dimensional input vector before being passed into the neural network.
        </p>
        <p>
        During training, the network attempts to correctly classify these digits by adjusting its weights and biases over multiple epochs.
        </p>

        <h2>Canvas</h2>
        <p>
        The canvas allows the user to manually draw a digit. Once drawn, the image is converted into a tensor and passed through the trained model.
        The predicted class and probabilities are then displayed in the output panel.
        This allows the user to test the trained network in real time.
        </p>

        <h2>Training Panel</h2>
        <p>
        The training panel allows the user to start, pause, or step through training one epoch at a time.
        Continuous training runs in the background, while step training allows controlled experimentation.
        The training size slider allows the user to limit how much of the dataset is used.
        </p>

        <h2>Hyperparameters</h2>
        <p>
        Hyperparameters control how the neural network behaves during training.
        Changing them will rebuild the model and reset training.
        </p>
        <ul>
        <li><b>Learning Rate</b> – Determines how large the weight updates are after each batch. A value that is too high may cause instability, while a value that is too low may result in slow learning.</li>
        <li><b>Batch Size</b> – Specifies how many samples are processed before updating the model parameters.</li>
        <li><b>Activation Function</b> – Introduces non-linearity into the network. Without activation functions, the network would behave like a simple linear model.</li>
        <li><b>Hidden Layers and Neurons</b> – Control the complexity and capacity of the network.</li>
        <li><b>Optimiser</b> – Determines how gradients are used to update the model parameters (e.g. SGD or Adam).</li>
        </ul>

        <h2>Network View</h2>
        <p>
        The network view visualises the structure of the neural network.
        Each node represents a neuron, and each connection represents a weight.
        The grayscale colour of each neuron reflects its activation value.
        Hovering over nodes and edges displays their numerical values.
        </p>

        <h2>Output Panel</h2>
        <p>
        The output panel displays the softmax probabilities for each digit class.
        This allows the user to see not only the predicted digit, but also how confident the network is in its decision.
        </p>

        <h2>Analytics</h2>
        <p>
        The analytics page displays training loss and accuracy over epochs.
        This allows the user to evaluate whether the model is learning effectively and whether the chosen hyperparameters are suitable.
        </p>

        <p>
        Overall, this application allows the user to both visualise and control the training process,
        providing a deeper understanding of how neural networks operate internally.
        </p>
        """)
        layout.addWidget(text)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

