# from Image_Classification_From_Scratch.old_backend.sequential import Sequential
# from Image_Classification_From_Scratch.old_backend.optimisers import *
# from Image_Classification_From_Scratch.old_backend.model import *
from device import DEVICE
# Modules from torch that are needed
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
from hyper_panel import HyperParams
# from Image_Classification_From_Scratch.old_backend.linear import Linear
# from Image_Classification_From_Scratch.old_backend.activation_functions import *

from PyQt6.QtCore import pyqtSignal

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


# @dataclass
# class NetworkSnapshot:
#     weights: list = List[torch.Tensor]
#     biases: list = List[torch.Tensor]
#     inputs: list = List[torch.Tensor] # in fact are the outputs attributes of the linear objects
#     activated_outputs: list = List[torch.Tensor] # output attribute of the activation function's objects
#     outputs: list = List[torch.Tensor]

SNAPSHOT = {
    "weights": [],
    "biases": [],
    "outputs": [],
    "activated_outputs": [],
    "inputs": []
}

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hyper_params = HyperParams()

        activation = self.hyper_params.activation

        if activation == "ReLU":
            self.network = nn.Sequential(
                nn.Linear(in_features=28 * 28, out_features=512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=512, out_features=10),
                nn.ReLU(),
            )

        elif activation == "Tanh":
            self.network = nn.Sequential(
                nn.Linear(in_features=28 * 28, out_features=512),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(in_features=512, out_features=10),
                nn.Tanh(),
            )

        elif activation == "Sigmoid":
            self.network = nn.Sequential(
                nn.Linear(in_features=28 * 28, out_features=512),
                nn.Sigmoid(),
                nn.Dropout(0.2),
                nn.Linear(in_features=512, out_features=10),
                nn.Sigmoid(),
            )


        # set the optimiser, learning rater and batch size from hyper_params
        # as class attributes
        self.optimiser = self.hyper_params.optimiser
        self.lr = self.hyper_params.lr
        self.batch_size = self.hyper_params.batch_size

        # These are used inside MainWindow for the NetworkView
        # stores the raw outputs of each linear layer
        self.last_linear_outputs = []
        # stores the outputs after activation func has been applied
        self.last_activation_outputs = []
        # stores the final output of the network
        self.last_logits = None


    def forward(self, inputs):
        # inputs = self.flatten(inputs)
        self.last_linear_outputs = []
        self.last_activation_outputs = []

        x = inputs

        for layer in self.network:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                self.last_linear_outputs.append(x)

            elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
                self.last_activation_outputs.append(x)

        self.last_logits = x

        return x

        # self.hyper_params = HyperParams()
        # self.snapshot = NetworkSnapshot()
        # self.snapshot = {
        #     "weights": [],
        #     "biases": [],
        #     "outputs": [],
        #     "activated_outputs": [],
        #     "inputs": []
        #
        # }
class Training:
    def __init__(self):
        # This is my new model
        self.model = NeuralNetwork().to(DEVICE)
        self.loss_fn = nn.CrossEntropyLoss()
        self.hyper_params = self.model.hyper_params

        self.optimiser = self.model.optimiser
        self.lr = self.hyper_params.lr

        if self.optimiser == "SGD":
            self.optimiser = torch.optim.SGD(params=self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        elif self.optimiser == "Adam":
            self.optimiser = torch.optim.Adam(params=self.model.parameters(), lr=1e-3, weight_decay=1e-4 )

        # We need to flatten the input data before we can pass it into the dense layers
        # For now keep it flattened but, when CNN implemented, change later
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])

        # Load MNIST dataset

        # training dataset
        self.train_dataset = datasets.MNIST(
            root= "./data",
            train=True,
            download=True,
            transform=transform
        )

        # testing dataset
        self.test_dataset = datasets.MNIST(
            root=  "./data",
            train=False,
            download=True,
            transform=transform
        )

        # Create batches of data
        self.batch_size = self.model.batch_size

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # activation= self.hyper_params.activation



        # model_features = [
        #     ("linear", 784, 128),
        #     ("activation", activation),
        #     ("linear", 128, 10),
        #     ("activation", activation),
        #     ("activation", "Softmax")
        # ]


        # self.network = Sequential(model_features, device=DEVICE)
        # self.network.create_instances()


        # self.optimiser = self.hyper_params.optimiser
        #
        # if self.optimiser == "SGD":
        #     self.optimiser = torch.optim.SGD(params=model.parameters(), lr=0.001)
        # else:
        #     self.optimiser = Adam(lr=0.1)

        # self.learning_rate = self.hyper_params.lr

        # self.model = Model(sequential=self.network, optimiser=self.optimiser, lr=self.learning_rate)
        # self.set_snapshot()

        self.curr_training_batch = None
        self.loss_values = []
        self.accuracy_values = []

        self.epoch_counter = 0

    def apply_hyperparams(self, config: HyperParams):
        # apply new hyperparameters from the UI.
        # rebuilds optimiser / network and recreates dataloaders when required.
        # call this only when training is paused.

        # dataclass that contains lr, batch_size, etc.
        self.hyper_params = config

        # Update batch size + recreate loaders if needed
        new_batch = int(getattr(config, "batch_size", self.batch_size))
        if new_batch != self.batch_size:
            self.batch_size = new_batch

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True
            )

            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )

        # Rebuild network if activation changed

        self.model = NeuralNetwork().to(DEVICE)
        self.loss_fn = nn.CrossEntropyLoss()

        self.optimiser = self.hyper_params.optimiser
        self.lr = self.hyper_params.lr

        if self.optimiser == "SGD":
            self.optimiser = torch.optim.SGD(params=self.model.parameters(), lr=1e-3, weight_decay=1e-4 )

        elif self.optimiser == "Adam":
            self.optimiser = torch.optim.Adam(params=self.model.parameters(), lr=1e-3, weight_decay=1e-4 )


        # activation = getattr(config, "activation", self.hyper_params.activation)
        # model_features = [
        #     ("linear", 784, 128),
        #     ("activation", activation),
        #     ("linear", 128, 10),
        #     ("activation", activation),
        #     ("activation", "Softmax")
        # ]
        #
        # self.network = Sequential(model_features, device=DEVICE)
        # self.network.create_instances()
        #
        # # Optimiser selection
        # opt_name = getattr(config, "optimiser", "SGD")
        # lr = float(getattr(config, "lr", self.learning_rate))
        # self.learning_rate = lr
        #
        # if opt_name == "SGD":
        #     optimiser = SGD(lr=0.1)
        # # else:
        # #     optimiser = Adam(lr=0.1)
        #
        # self.optimiser = optimiser
        # self.model = Model(sequential=self.network, optimiser=self.optimiser, lr=self.learning_rate)



        # Clear any stale snapshot state
        self.clear_snapshot()
        self.curr_training_batch = None

    def apply_new_training_size(self, new_training_size: int):
        subset = Subset(self.train_dataset, range(new_training_size))
        self.train_loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
        )


    def set_snapshot(self):
        SNAPSHOT["inputs"].append(None if self.curr_training_batch is None else self.curr_training_batch.detach().cpu())

        SNAPSHOT["weights"].clear()
        SNAPSHOT["biases"].clear()
        for layer in self.model.network:
            if isinstance(layer, nn.Linear):
                SNAPSHOT["weights"].append(layer.weight.detach().cpu())
                SNAPSHOT["biases"].append(layer.bias.detach().cpu())

        # SNAPSHOT["outputs"] = [t.detach().cpu() for t in getattr(self.model, "last_linear_outputs", [])]
        # SNAPSHOT["activated_outputs"] = [t.detach().cpu() for t in getattr(self.model, "last_activation_outputs", [])]
        # Outputs / activations captured from the most recent forward pass
        linear_outs = [t.detach().cpu() for t in getattr(self.model, "last_linear_outputs", [])]
        act_outs = [t.detach().cpu() for t in getattr(self.model, "last_activation_outputs", [])]

        # The UI typically treats each weight matrix as a "layer" transition.
        # So we ensure outputs has one entry per Linear layer.
        n_linear = len(SNAPSHOT["weights"])

        # Pad / trim linear outputs to exactly n_linear
        if len(linear_outs) < n_linear:
            linear_outs = linear_outs + [None] * (n_linear - len(linear_outs))
        else:
            linear_outs = linear_outs[:n_linear]

        # Activations exist after each hidden Linear (not after final logits),
        # but the UI indexes activated_outputs with (layer_i - 1), which can hit the output layer too.
        # So we pad to n_linear as well, leaving the final activation as None.
        if len(act_outs) < n_linear:
            act_outs = act_outs + [None] * (n_linear - len(act_outs))
        else:
            act_outs = act_outs[:n_linear]

        SNAPSHOT["outputs"] = linear_outs
        SNAPSHOT["activated_outputs"] = act_outs




    def clear_snapshot(self):
        SNAPSHOT["weights"].clear()
        SNAPSHOT["biases"].clear()
        SNAPSHOT["outputs"].clear()
        SNAPSHOT["activated_outputs"].clear()
        SNAPSHOT["inputs"].clear()

    def one_hot(self, labels, num_classes=10, device=DEVICE):
        y = torch.zeros(labels.size(0), num_classes, device=device, dtype=torch.int64)
        y.scatter_(1,labels.unsqueeze(1),1)
        return y

    def training(self):
        # Training
        # epochs = 2
        # self.set_snapshot()
        # print(f"weights: {len(SNAPSHOT["weights"])} \n{SNAPSHOT['weights']}")
        # print(f"inputs: {len(SNAPSHOT["inputs"])} \n{SNAPSHOT['inputs']}")
        # print(f"outputs: {len(SNAPSHOT["outputs"])} \n{SNAPSHOT['outputs']}")
        # print(f"activated_outputs: {len(SNAPSHOT["activated_outputs"])} \n{SNAPSHOT['activated_outputs']}")

        # for epoch in range(epochs):
        self.model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for training_data, sample_labels in self.train_loader:
            training_data = training_data.to(DEVICE)
            sample_labels = sample_labels.to(DEVICE)

            # y_true = self.one_hot(sample_labels, 10, DEVICE)

            self.curr_training_batch = training_data

            logits = self.model(training_data)
            loss = self.loss_fn(logits, sample_labels)

            self.optimiser.zero_grad(set_to_none=True)
            loss.backward()
            self.optimiser.step()

            batch_size = sample_labels.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == sample_labels).sum().item()
            total_samples += batch_size

        if total_samples == 0:
            avg_loss = 0
            acc = 0

        else:
            avg_loss = total_loss / total_samples
            acc = (total_correct / total_samples) * 100

        self.clear_snapshot()
        self.set_snapshot()
        self.epoch_counter += 1

        self.loss_values.append(avg_loss)
        self.accuracy_values.append(acc)

        # self.epochTrained.emit()

        return (avg_loss, acc)




    # # Testing
    # def testing(self):
    #     self.model.model_structure.to(device=DEVICE)
    #
    #     test_total_loss = 0
    #     test_correct = 0
    #     test_total = 0
    #
    #     for inputs, labels in self.test_loader:
    #         inputs = inputs.to(DEVICE)
    #         labels = labels.to(DEVICE)
    #
    #         pred = self.model.model_structure.forward(inputs)
    #
    #         pred_labels = pred.argmax(dim=1)
    #         test_correct += (pred_labels == labels).sum().item()
    #         test_total += labels.size(0)
    #
    #         y_true = self.one_hot(labels, 10, DEVICE)
    #         batch_loss = self.model.cross_entropy.forward(pred, y_true)
    #         test_total_loss += batch_loss.item()
    #
    #     self.clear_snapshot()
    #     self.set_snapshot()
    #
    #     test_avg_loss = test_total_loss / len(self.test_loader)
    #     test_acc = (test_correct / test_total) * 100
    #     print(f"Test Loss: {test_avg_loss}, Test Accuracy: {test_acc}")


if __name__ == "__main__":
    training = Training()
    training.training()
    training.testing()


