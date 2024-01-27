import numpy as np
from tqdm import tqdm

from losses import Loss
from activations import Function
from layer import HiddenLayer
from regularizers import Regularizer
from utils import Starting_values
from metrics import Metric


class Network:
    """Class that represents a neural network. It has a variable number of
    hidden layers, a loss function used to calculate the error between the
    produced output and the expected output, and a number of hyperparameters
    that control the training algorithm.
    """

    def __init__(self, loss: Loss, eta: float = 1e-1, tau: int = 1000, cyclic: bool = False):
        """Initialize the network.
        The network is initialized with an empty list of layers and a loss
        function, as well as eta (the learning rate), tau (learning rate decay
        hyperparameter), and cyclic (which controls whether to use cyclic
        learning rate).
        """

        self.layers = []
        self.loss = loss

        self.inputs = None
        self.eta = eta
        self.tau = tau
        self.eta_tau = eta * 0.01
        self.cyclic = cyclic
        self.n_cycles = 0 # counter for the number of cycles
        self.cycle_flag = False
        self.cycles = 1

    def add_layer(
            self,
            input_size,
            units_size,
            starting: Starting_values,
            activation: Function,
            regularizer: Regularizer = None,
            momentum: tuple[str, float] = None
    ):
        """Add a layer with the specified parameters to the network."""
        if len(self.layers) > 0:
            self.check_layers_shape(self.layers[-1].units_size, input_size)

        self.layers.append(
            HiddenLayer(
                input_size=input_size,
                units_size=units_size,
                starting=starting,
                activation=activation,
                regularizer=regularizer,
                momentum=momentum
            )
        )

    def forward(self, inputs: np.ndarray):
        """Calculate the output of the neural network by forwarding the inputs
        through the layers.
        """
        self.inputs = inputs

        o = self.layers[0].forward(inputs)
        for layer in self.layers[1:]:
            o = layer.forward(o)

        return o

    def backward(self, curr_delta: np.ndarray, step: int):
        """Backpropagate the error through the network."""

        # Adjust the learning rate of the network
        if self.cyclic:
            # cyclical learning rate non-smooth
            """ 
            if step % self.tau == 0:
                if not self.cycle_flag:
                    self.cycle_flag = True
                    self.n_cycles += 1
                    print(f"Cycle {self.n_cycles} completed.")
            else:
                self.cycle_flag = False
            step = step % self.tau # cycle of tau = 1000
            eta_step = self.eta_tau + 0.5 * (self.eta / 2**self.n_cycles - self.eta_tau) * (1 + np.cos(np.pi * step / self.tau))
            """

            # cyclical learning rate smooth
            if step % self.tau == 0:
                if not self.cycle_flag:
                    self.cycle_flag = True
                    self.cycles *= -1
                    if self.cycles == -1:
                        self.n_cycles += 1
            else:
                self.cycle_flag = False
            step = step % self.tau
            eta_step = self.eta_tau + 0.5 * (self.eta / 2 ** self.n_cycles - self.eta_tau) * (1 + np.cos(np.pi * step / self.tau)) * self.cycles
            if self.cycles == -1:
                eta_step = eta_step + self.eta / 2 ** self.n_cycles - self.eta_tau
        else:
            # Linear decay of eta
            if step > self.tau:
                step = self.tau
            alpha = step / self.tau
            eta_step = (1 - alpha) * self.eta + alpha * self.eta_tau

        for layer in reversed(self.layers):
            delta_prop = layer.backward(curr_delta, eta_step)
            curr_delta = delta_prop

    def check_layers_shape(self, units_size: int, input_size: int):
        """Check whether the provided sizes are equal, raising an exception if not."""
        if units_size != input_size:
            raise ValueError(
                f"Error: vectors must have same shape."
                f"The shapes are units_size: {units_size} and input_size: {input_size}"
            )

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            metric: Metric,
            epochs: int,
            patience: int = np.inf,
            thresh: float = 0.01,
            final_retrain=False,
            final_tr_loss=0,
    ):
        """Train the neural network on the provided training data. Losses and
        metrics are calculated for each epoch (both for training and
        validation sets), and then returned as part of a single dictionary.
        """

        tr_losses = []
        tr_metrics = []
        val_losses = []
        val_metrics = []

        epochs_since_lowest = 0

        best_W = []
        best_b = []
        for layer in self.layers:
            best_W.append(layer.W.copy())
            best_b.append(layer.b.copy())

        with tqdm(total=epochs, desc="Epochs", colour='yellow') as pbar:
            lowest = np.inf  # Variable that stores the lowest loss recorded

            for epoch in range(epochs):
                # Training the network
                for X, y in zip(X_train, y_train):
                    out = self.forward(inputs=X)
                    self.backward(self.loss.backward(y_pred=out, y_true=y), epoch + 1)
                    """if k >= 4:
                        for layer in self.layers:
                            print("W:\n",layer.W)
                            print("b:\n",layer.b)
                        print("out:\n",out)
                        print("y:\n",y)
                        print("loss deriv:\n",self.loss.backward(y_pred=out, y_true=y))"""

                    # togliere fine debug @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

                    # print(self.layers[0], end='\r')
                    # time.sleep(0.1)

                    # togliere fine debug @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

                # Calculating loss and accuracy for the epoch
                # Training loss and acc
                tr_loss, tr_acc = self.statistics(X_train, y_train, metric)
                tr_losses.append(tr_loss)
                tr_metrics.append(tr_acc)

                # Validation loss and acc
                val_loss, val_acc = self.statistics(X_val, y_val, metric)
                val_losses.append(val_loss)
                val_metrics.append(val_acc)

                if final_retrain:
                    # If performing final retrain on all data (checking if tr
                    # loss reaches mean observed during CV)
                    if tr_loss < final_tr_loss:
                        for layer in self.layers:
                            best_W.append(layer.W.copy())
                            best_b.append(layer.b.copy())
                        break
                else:
                    # Check early stopping condition
                    if val_loss < (lowest - lowest * thresh) or lowest == np.inf:
                        lowest = val_loss
                        epochs_since_lowest = 0

                        # Record all the layers' weights and bias that correspond to
                        # the lowest validation loss
                        best_W = []
                        best_b = []
                        for layer in self.layers:
                            best_W.append(layer.W.copy())
                            best_b.append(layer.b.copy())
                    else:
                        epochs_since_lowest = epochs_since_lowest + 1

                    if epochs_since_lowest >= patience:
                        # Early stopping cond is true
                        break

                pbar.update(1)

        # Network is set with the weights/bias associated with the lowest
        # validation loss
        for i, layer in enumerate(self.layers):
            layer.W = best_W[i]
            layer.b = best_b[i]

        statistics = {
            'tr_losses': tr_losses,
            'tr_metrics': tr_metrics,
            'val_losses': val_losses,
            'val_metrics': val_metrics
        }

        return statistics

    def statistics(self, X, y, metric: Metric, scaler=None):
        """Calculate loss and metric for the given input and output data."""
        pred = []
        for x in X:
            out = self.forward(x)
            pred.append(out)
        pred = np.array(pred)

        if scaler:
            pred = pred.reshape(pred.shape[0], pred.shape[1])
            pred = scaler.inverse_transform(pred)
            pred = pred.reshape(pred.shape[0], pred.shape[1], 1)

            y_rescaled = y.copy()
            y_rescaled = y_rescaled.reshape(y_rescaled.shape[0], y_rescaled.shape[1])
            y_rescaled = scaler.inverse_transform(y_rescaled)
            y_rescaled = y_rescaled.reshape(y_rescaled.shape[0], y_rescaled.shape[1], 1)

            loss = self.loss.forward(y_pred=pred, y_true=y_rescaled)
            metric_eval = metric(y_pred=pred, y_true=y_rescaled)
        else:
            loss = self.loss.forward(y_pred=pred, y_true=y)
            metric_eval = metric(y_pred=pred, y_true=y)

        return loss, metric_eval

    def __str__(self) -> str:
        """Print the network information."""
        return f"Network: {len(self.layers)} layers \nLoss: {self.loss}"

    def to_csv(self):
        return (
                f"\n{self.loss}\n"
                f"eta: {self.eta}\n"
                f"tau: {self.tau}\n"
                f"eta_tau: {self.eta_tau}"
                f"cyclic: {self.cyclic}\n"
                )