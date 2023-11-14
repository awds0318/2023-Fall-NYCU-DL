import numpy as np


class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *output_grad):
        raise NotImplementedError


class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))

    def forward(self, input):
        self.input = input
        output = np.matmul(input, self.weight) + self.bias
        return output

    def backward(self, output_grad):
        input_grad = np.dot(output_grad, self.weight.T)
        self.weight_grad = np.dot(self.input.T, output_grad)
        self.bias_grad = np.sum(output_grad, axis=0)

        return input_grad

    def update(self, lr):
        self.weight -= lr * self.weight_grad
        self.bias -= lr * self.bias_grad


class Activation(_Layer):
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        # ReLU
        output = np.maximum(0, input)
        return output

    def backward(self, output_grad):
        input_grad = output_grad
        input_grad[self.input < 0] = 0
        return input_grad


class SoftmaxWithloss(_Layer):
    def __init__(self):
        self.y = None
        self.y_pred = None

    def forward(self, input, y):
        self.input = input
        self.y = y
        # Softmax
        self.y_pred = softmax(input)
        # CE loss
        loss = cross_entropy_loss(self.y_pred, y)

        return self.y_pred, loss

    def backward(self):
        input_grad = (self.y_pred - self.y) / self.y_pred.shape[0]

        return input_grad


def softmax(x):
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y):
    size = y_pred.shape[0]
    y_pred = y_pred.clip(min=1e-8, max=None)
    loss = -np.sum(np.log(y_pred) * y) / size
    return loss
