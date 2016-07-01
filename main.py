import numpy as np

def sigmoid(x, deriv = False):
    if deriv:
        return sigmoid(x)* (1 - sigmoid(x))
    return 1/(1 + np.exp(-x))

class NeuralNetwork:

    class NeuralLayer:

        class Neuron:
            def __init__(self, syn_count, bias):
                self.bias = bias
                self.weights = [np.random.randn(i, 1) for i in range(syn_count)]

            def activate(self, x):
                return sigmoid(np.dot(x, self.weights) + self.bias)

        def __init__(self, count, syn_per_neuron):
            biases = [np.random.randn(i, 1) for i in range(count)]
            self.neurons = [self.Neuron(syn_per_neuron, biases[i]) for i in range(count)]

        def feed_forward(self, inputs):
            return [self.neurons[i].activate(inputs[i]) for i in len(self.neurons)]

    def __init__(self, sizes):
        self.layers = [self.NeuralLayer(sizes[i],3) for i in range(len(sizes))]

    def feed_forward(self, x):
        """

        :param x: inputs
        :return: output on inputs
        """

        pass

jimmy = NeuralNetwork([2,2,1])