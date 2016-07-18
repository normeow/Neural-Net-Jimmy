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
                self.weights = np.random.randn(syn_count, 1)

            def activate(self, x):
                s = sigmoid(np.dot(x, self.weights) + self.bias)
                return s


        def __init__(self, bias, count, syn_per_neuron):
            self.neurons = [self.Neuron(syn_per_neuron, bias) for i in range(count)]

        def feed_forward(self, inputs):
            self.inputs = inputs
            self.outputs = [n.activate(self.inputs) for n in self.neurons]

    def __init__(self, sizes, learning_rate = 0.5):
        pass
        #self.layers = [self.NeuralLayer(bias, sizes[i],) for i in range(len(sizes))]


    def feed_forward(self, x):
        o = self.layers[0].feed_forward(x)
        for layer in self.layers[1:]:
            o = layer.feed_forward(o)
        return o

    def train(self, inputs, answers):
        for i in range(len(inputs)):
            output = self.feed_forward(inputs[i])
            q_errors = np.dot(0.5*(answers[i] - output))


jimmy = NeuralNetwork([2,2,1])
X = np.array([[0, 1],
              [1, 0],
              [0, 0],
              [1, 1]])
# output matrix
Y = np.array([[1, 1, 0, 0]])
#jimmy.train(X,Y.T)