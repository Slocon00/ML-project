class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError("Loss.forward() not implemented.")

    def backward(self):
        raise NotImplementedError("Loss.backward() not implemented.")
