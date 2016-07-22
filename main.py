import numpy as np

def sigmoid(x, deriv = False):
    if deriv:
        return sigmoid(x)* (1 - sigmoid(x))
    return 1/(1 + np.exp(-x))

class NeuralNetwork:

    class NeuralLayer:

        class Neuron:
            def __init__(self, syn_count, bias, learning_rate):
                self.learning_rate = learning_rate
                self.bias = bias
                self.weights = np.random.randn(syn_count)
                self.input = 0
                self.weight_sum = 0

            def activate(self, x):
                self.input = x
                self.weight_sum = np.dot(x, self.weights) + self.bias
                self.output = sigmoid(self.weight_sum)
                return self.output

            def calculate_err(self, weight_deltas):
                self.delta = sigmoid(self.weight_sum)*sum(weight_deltas)
                return self.delta*self.weights

            def reweight(self):
                pass


        def __init__(self, count, syn_per_neuron, learning_rate):
            self.inputs = []
            self.outputs = []
            self.deltas = None
            self.bias = np.random.sample()
            self.neurons = [self.Neuron(syn_per_neuron, self.bias, learning_rate) for i in range(count)]

        def feed_forward(self, inputs):
            self.inputs = inputs
            self.outputs = [n.activate(self.inputs) for n in self.neurons]
            return self.outputs

        def calculate_errs(self, weight_prev_deltas):
            pre_d = [self.neurons[i].calculate_err(weight_prev_deltas[i]) for i in range(len(self.neurons))]
            self.deltas = np.array(pre_d).T

        def reweight(self):
            pass

    def __init__(self, inputs_count, sizes, learning_rate = 0.5):
        self.layers = []
        self.sizes = sizes
        for i in range(len(sizes)):
            if i == 0:
                self.layers.append(self.NeuralLayer(sizes[i], inputs_count, learning_rate))
            else:
                self.layers.append(self.NeuralLayer(sizes[i], sizes[i-1], learning_rate))

    def feed_forward(self, x):
        o = self.layers[0].feed_forward(x)
        for layer in self.layers[1:]:
            o = layer.feed_forward(o)
        return o

    def backprop(self, error):
        #last layer calculate its weighted deltas
        # короч, это неправильно - когда обрабатываем последний слой нужно передавать ошибку *  количество нейронов в предпоследнем слое
        deltas_o = self.layers[-1].calculate_errs(error*self.sizes[-1])

        for layer in self.layers:
            pass

    def train(self, inputs, answers):
        for i in range(len(inputs)):
            output = self.feed_forward(inputs[i])
            error  = answers[i] - output
            q_error = 0.5*sum(error)
            self.backprop(error)


jimmy = NeuralNetwork(2, [3,2])
X = np.array([[0, 1],
              [1, 0],
              [0, 0],
              [1, 1]])
# output matrix
Y = np.array([[0, 1],
              [1, 0],
              [0, 0],
              [1, 1]])
print(np.dot([2,1,3], [-2,1,4]) + 4)
jimmy.train(X,Y)