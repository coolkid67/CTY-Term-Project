import numpy as np

def sigmoid(x):
    #f(x) = 1 / (1+e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

weights = np.array([0, 1])
bias = 0 # temporary, doesn't matter
n = Neuron(weights, bias)

x = np.array([1, -1]) # 1 = ptera, 0 = ground, -1 = cactus
print(n.feedforward(x))