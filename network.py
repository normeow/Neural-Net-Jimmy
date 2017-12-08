import numpy as np
import dill

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
                self.inputs = 0
                self.weight_sum = 0

            def activate(self, x):
                self.inputs = x
                self.weight_sum = np.dot(x, self.weights) + self.bias
                self.output = sigmoid(self.weight_sum)
                return self.output

            def calculate_err(self, weight_deltas, outp = False):
                '''
                :param weight_deltas:
                :return: weighted delta
                '''
                self.delta = sigmoid(self.weight_sum, True)*sum(weight_deltas)
                if outp:
                    self.delta *= -1
                return self.delta*self.weights

            def reweight(self):
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate*self.delta*self.inputs[i]


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

        def calculate_errs(self, weight_prev_deltas, outp = False):
            '''
            :param weight_prev_deltas:
            :return: matrix of weighed deltas
            '''
            pre_d = [self.neurons[i].calculate_err(weight_prev_deltas[i], outp) for i in range(len(self.neurons))]
            self.deltas = np.array(pre_d).T
            return  self.deltas

        def reweight(self):
            for neuron in self.neurons:
                neuron.reweight()

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
        errs = [[error[i]] for i in range(len(error))]
        deltas_o = self.layers[-1].calculate_errs(errs, True)

        for layer in self.layers[-2::-1]:
            deltas_o = layer.calculate_errs(deltas_o)

        for layer in self.layers:
            layer.reweight()

    def predict(self, x):
        res = []
        for i in x:
            res.append(self.feed_forward(i))
        return res


    def train(self, inputs, answers):

        for j in range(100000):
            for i in range(len(inputs)):
                output = self.feed_forward(inputs[i])
                error  = answers[i] - output
                q_egitrror = 0.5*sum(error)
                if j % 10000 == 0:
                    print(output)
                self.backprop(error)
            if j % 10000 == 0:
                print()

    def save_model(self, path):
        with open(path, 'w+b') as f:
            dill.dump(self, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
             self = dill.load(f)


