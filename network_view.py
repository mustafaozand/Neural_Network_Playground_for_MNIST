from PyQt6.QtCore import Qt, QPointF, QRectF, QSize
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor
from PyQt6.QtWidgets import QMainWindow, QFrame, QToolTip, QWidget, QVBoxLayout, QApplication
import sys
# from training import NetworkSnapshot
from training import SNAPSHOT
import torch
from training import Training
import math

class NetworkView(QFrame):
    def __init__(self, ):
        super().__init__()

        self.setObjectName("NetworkView")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(QSize(600, 300))
        # self.setMaximumSize(QSize(800, 600))

        self.setMouseTracking(True)

        self.margin = 24
        self.node_rad = 24
        self.node_gap = 48
        self.edge_hit_threshold = 6
        self.padding = 24

        # self.trainer = Training()
        # self.trainer.training()

        self.snapshot = SNAPSHOT

        self.node_hitboxes = {}
        self.edge_hitboxes = {}
        self.x_pos = []
        self.y_pos = []
        self.no_layers = None
        self.no_nodes = None
        self.gray_scale_values = []

        self.last_hover_key = None

        self.displayed_values = {
            "weights": [],
            "biases": [],
            "outputs": [],
            "activated_outputs": [],
            "inputs": []
        }

        for id in self.displayed_values:
            arr = self.snapshot[id]
            self.partition_data(arr, id)

        print(self.displayed_values)

        if not self.displayed_values["biases"]:
            dummy = self.make_dummy_snapshot(layers=4, num_nodes=5)  # choose layers here
            for key in self.displayed_values:
                self.partition_data(dummy[key], key)

    def make_dummy_snapshot(self, layers, num_nodes):
        num_layers = layers - 1

        snap = {
            "inputs": [torch.rand(num_nodes)],

            "biases": [torch.randn(num_nodes) for _ in range(num_layers)],

            "weights": [torch.randn(num_nodes, num_nodes) for _ in range(num_layers)],

            "outputs": [torch.randn(num_nodes) for _ in range(num_layers)],

            "activated_outputs": [torch.rand(num_nodes) for _ in range(num_layers)],
        }
        return snap

    def set_snapshot(self, snapshot_dict):
        self.snapshot = snapshot_dict

        for key in self.displayed_values:
            self.partition_data(self.snapshot.get(key, []), key)

        self.update()

        #
        # for tensor in arr:
        #     partitioned = torch.cat((tensor[0][:3], tensor[0][-2:]), dim=0)
        #     new_arr.append(partitioned)
        #
        # self.displayed_values[dict_id] = new_arr

    def partition_data(self, arr, dict_id:str):
        new_arr = []

        for tensor in arr:
            if tensor is None:
                new_arr.append(None)

            elif dict_id == "weights" and tensor.dim() == 2:
                # tensor shape: (out_neurons,in_features)
                partitioned = torch.cat((tensor[:3], tensor[-2:]), dim=0)
                partitioned = torch.cat((partitioned[:, :3], partitioned[:, -2:]), dim=1)
                new_arr.append(partitioned)
                continue

            elif tensor.dim() == 1:
                # (neurons,) like biases
                partitioned = torch.cat((tensor[:3], tensor[-2:]), dim=0)
                new_arr.append(partitioned)

            else:
                # (batch,neurons) like outputs,activated_outputs,inputs
                partitioned = torch.cat((tensor[0][:3], tensor[0][-2:]), dim=0)
                new_arr.append(partitioned)

        self.displayed_values[dict_id] = new_arr

    def compute_display_layout(self):
        self.x_pos.clear()
        self.y_pos.clear()

        w = self.width()
        h = self.height()

        # top left is (0,0)
        left = self.margin
        top = self.margin
        right = w - self.margin
        bottom = h - self.margin

        # number of layers

        self.no_layers = len(self.displayed_values["outputs"]) + 1

        usable_w = right - left
        usable_h = bottom - top

        # x_positions = [left + (usable_w * i) / (no_layers-1) for i in range(no_layers)]

        for x in range(self.no_layers):
            self.x_pos.append(left + (usable_w * x) / (self.no_layers + 1) + (usable_w / (self.no_layers + 1)))

        self.no_nodes = len(self.displayed_values["biases"][0])

        for y in range(self.no_nodes):
            self.y_pos.append((top + (usable_h * y) / (self.no_nodes+1)) + self.padding * 4)

    def draw_nodes(self, painter:QPainter):

        r = self.node_rad

        node_pen = QPen(Qt.GlobalColor.black)
        node_pen.setWidth(1)

        node_brush = QBrush(Qt.GlobalColor.white)

        ellipses_pen = QPen(Qt.GlobalColor.darkGray)
        ellipses_pen.setWidth(1)

        gray_scale = self.compute_grayscale_values()

        for layer in range(self.no_layers):
            for n in range(self.no_nodes):
                painter.setPen(node_pen)

                if layer != 0 and gray_scale is not None:
                    value = gray_scale[layer-1][n]
                    value = int(round(float(value)))
                    value = max(0, min(255, value))
                    color = QColor(value, value, value)
                    node_brush = QBrush(color)

                painter.setBrush(node_brush)

                rect = QRectF(self.x_pos[layer] - r, self.y_pos[n] - r, 2 *r, 2*r)
                painter.drawEllipse(rect)

                self.node_hitboxes[layer, n] = rect

    def draw_edges(self, painter: QPainter):

        pen = QPen(Qt.GlobalColor.lightGray)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for layer in range(self.no_layers -1):
            for n1 in range(self.no_nodes):
                for n2 in range(self.no_nodes):
                    x1 = int(round(self.x_pos[layer]))
                    y1 = int(round(self.y_pos[n1]))
                    x2 = int(round(self.x_pos[layer + 1]))
                    y2 = int(round(self.y_pos[n2]))

                    painter.drawLine(x1, y1, x2, y2)

                    a = QPointF(x1, y1)
                    b = QPointF(x2, y2)

                    self.edge_hitboxes[(layer, n1, n2)] = (a, b)

    def calc_distance_between_cursor_and_edge(self, point: QPointF, a:QPointF, b:QPointF):
        ax, ay = a.x(), a.y()
        bx, by = b.x(), b.y()
        px, py = point.x(), point.y()

        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay

        # Length of AB squared
        ab_len2 = abx ** 2 + aby ** 2

        # if ab_len2 = 0 then points a and b are equal

        if ab_len2 <= 1e-9:
            return math.hypot(px -ax, py-ay)

        # take the dot product of AP and AB
        # t tells us how far along AB the  projection of P falls
        t = (apx * abx + apy * aby) / ab_len2
        t = max(0.0, min(1.0, t))

        # Closest point on the line to the mouse point
        cx = ax + t * abx
        cy = ay + t * aby

        return math.hypot(px - cx, py - cy)

    def node_tooltip(self, layer_i, neuron_i):
        nodes = [f"Layer {layer_i} Neuron {neuron_i}"]

        if self.displayed_values["activated_outputs"][layer_i-1] is None:
            activated_value = None
            nodes.append(f"Activated value: {activated_value}")
            value = None
            nodes.append(f"Value: {value}")

            if layer_i != 0:
                biases = self.displayed_values["biases"][layer_i - 1][neuron_i]
                nodes.append(f"Biases: {biases}")

            return "\n".join(nodes)

        elif layer_i != 0:
            activated_value = self.displayed_values["activated_outputs"][layer_i-1][neuron_i]
            nodes.append(f"Activated value: {activated_value}")
            biases = self.displayed_values["biases"][layer_i-1][neuron_i]
            nodes.append(f"Biases: {biases}")
            value = self.displayed_values["outputs"][layer_i-1][neuron_i]
            nodes.append(f"Value: {value}")

        elif layer_i == 0:
            if self.displayed_values["inputs"][layer_i][neuron_i] is None:
                nodes.append(f"Inputs: {None}")

            else:
                inputs = self.displayed_values["inputs"][layer_i][neuron_i]
                nodes.append(f"Inputs: {inputs}")


        return "\n".join(nodes)

    def edge_tooltip(self, layer_from, from_neuron, to_neuron):

        weight = self.displayed_values["weights"][layer_from][to_neuron][from_neuron]

        lines = [f"Weight (L{layer_from}:{from_neuron} -> L{layer_from +1}:{to_neuron}"]

        lines.append(f"w = {weight}")

        return "\n".join(lines)

    def hit_test(self, curr_pos: QPointF):

        for (layer_i , neuron_i), rect in self.node_hitboxes.items():
            if rect.contains(curr_pos):
                return ("node", (layer_i, neuron_i), self.node_tooltip(layer_i, neuron_i))

        best = None
        for (layer_from, from_neuron, to_neuron), (a,b) in self.edge_hitboxes.items():
            d = self.calc_distance_between_cursor_and_edge(curr_pos, a, b)

            if d <= self.edge_hit_threshold:
                if best is None or d < best[0]:
                    best = (d, (layer_from, from_neuron, to_neuron))

        if best is not None:
            layer_from , from_neuron, to_neuron = best[1]
            return("edge", (layer_from, from_neuron, to_neuron), self.edge_tooltip(layer_from, from_neuron, to_neuron))

        return None

    def mouseMoveEvent(self, event):

        pos = event.position()

        hit = self.hit_test(pos)

        if hit is None:
            if self.last_hover_key is not None:
                QToolTip.hideText()
                self.last_hover_key = None
            return

        kind, key_tuple, text = hit
        hover_key = (kind, key_tuple)

        if hover_key != self.last_hover_key:
            QToolTip.showText(self.mapToGlobal(pos.toPoint()), text, self)
            self.last_hover_key = hover_key

    def leaveEvent(self, event):
        QToolTip.hideText()
        self.last_hover_key = None
        super().leaveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing,True)

        self.node_hitboxes.clear()
        self.edge_hitboxes.clear()

        self.compute_display_layout()

        self.draw_edges(painter)
        self.draw_nodes(painter)



        print(f"Hitboxes: {self.node_hitboxes}")

    def compute_grayscale_values(self):
        self.gray_scale_values.clear()

        activations = self.displayed_values["activated_outputs"]

        for layer_a in activations:
            if layer_a is None:
                self.gray_scale_values.append([0.0] * self.no_nodes)
                continue

            max_a = float(layer_a.max().item())
            if max_a == 0.0:
                layer_gray = [0.0 for _ in range(layer_a.numel())]
            else:
                layer_gray = [(float(v.item()) / max_a) * 255.0 for v in layer_a]

            self.gray_scale_values.append(layer_gray)

        return self.gray_scale_values


# if layer_a.dim() == 2:
#     layer_a = layer_a[0]
# if layer_a.dim() != 1:
#     layer_a = layer_a.reshape(-1)

class NetworkViewApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Network View")
        self.setGeometry(0, 0, 800, 600)

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        layout = QVBoxLayout(self.centralWidget)   # attach layout to centralWidget

        self.networkView = NetworkView()
        layout.addWidget(self.networkView)

# Demo Application
def main():
    app = QApplication(sys.argv)
    window = NetworkViewApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

