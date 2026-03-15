from PyQt6.QtWidgets import QWidget, QApplication, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import sys

class Analytics(QWidget):
    def __init__(self, trainer):
        super().__init__()
        # I use a vertical layout because the analytics view is basically a single page
        # where the graph canvas and its toolbar stack on top of each other.
        layout = QVBoxLayout()
        self.setLayout(layout)
        # Matplotlib uses a Figure object as the container for all graphs.
        # I have created one shared figure so that I can draw both the loss and accuracy plots in the same widget.
        self.fig = Figure()
        # I have split the figure into 2 rows so that loss is on top and accuracy is below.
        # This keeps the analytics readable and stops the graphs from overlapping.
        self.loss_ax = self.fig.add_subplot(2,1,1)
        self.accuracy_ax = self.fig.add_subplot(2,1,2)
     # The analytics page does not train the model itself.
        # Instead it reads values from the trainer so the training stage stays separate from the UI.
        self.trainer = trainer
        # self.trainer.epochTrained.connect(self.on_epoch_trained)

        # FigureCanvasQTAgg wraps the matplotlib figure so it can be embedded inside a pyQt widget.
        # This is what actually lets the graph appear in the GUI.
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)

        # The toolbar gives basic matplotlib controls like zoom and pan, which is useful for inspection.
        toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(toolbar)

        # I have stored references to the trainer's metric lists so the analytics view can plot
        # whatever has been collected so far.
        self.loss_values = self.trainer.loss_values
        self.accuracy_values = self.trainer.accuracy_values

    def on_epoch_trained(self):
        self.loss_values = self.trainer.loss_values
        self.accuracy_values = self.trainer.accuracy_values
        self.plot()

    def plot(self):
        self.loss_ax.clear()
        self.accuracy_ax.clear()

        epochs_x = list(range(1, self.trainer.epoch_counter + 1))

        # plot Loss
        self.loss_ax.plot(epochs_x, self.loss_values, label="Training Loss")
        # self.loss_ax.set_title("Training Loss vs Epoch")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.legend()

        # plot Accuracy
        self.accuracy_ax.plot(epochs_x, self.accuracy_values, label="Training Accuracy")
        # self.accuracy_ax.set_title("Training Accuracy vs Epoch")
        self.accuracy_ax.set_xlabel("Epoch")
        self.accuracy_ax.set_ylabel("Accuracy")
        self.accuracy_ax.legend()

        self.canvas.draw_idle()


# Demo Application
if __name__ == "__main__":

    class DummyTrainer:
        def __init__(self):
            # Dummy metric values for testing
            self.loss_values = [2, 1, 0.5, 0.3]
            self.accuracy_values = [30, 40, 60, 80]
            self.epoch_counter = 4

    app = QApplication(sys.argv)

    dummy_trainer = DummyTrainer()
    analytics = Analytics(dummy_trainer)

    # Plot immediately using dummy values
    analytics.plot()

    analytics.setWindowTitle("Analytics Dummy Demo")
    analytics.resize(600, 500)
    analytics.show()

    sys.exit(app.exec())