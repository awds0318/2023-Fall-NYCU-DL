from .layer import *


class Network(object):
    def __init__(self):
        self.fc1 = FullyConnected(28 * 28, 256)
        self.act1 = Activation()
        self.fc2 = FullyConnected(256, 128)
        self.act2 = Activation()
        self.fc3 = FullyConnected(128, 64)
        self.act3 = Activation()
        self.fc4 = FullyConnected(64, 10)
        self.loss = SoftmaxWithloss()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self, input, y):
        h1 = self.fc1.forward(input)
        h1 = self.act1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.act2.forward(h2)
        h3 = self.fc3.forward(h2)
        h3 = self.act3.forward(h3)
        h4 = self.fc4.forward(h3)
        pred, loss = self.loss.forward(h4, y)

        return pred, loss

    def backward(self):
        dloss = self.loss.backward()
        dh4 = self.fc4.backward(dloss)
        dh3 = self.act3.backward(dh4)
        dh3 = self.fc3.backward(dh3)
        dh2 = self.act2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.act1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update(lr)
