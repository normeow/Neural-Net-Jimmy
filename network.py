import numpy as np
import dill


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:

    def __init__(self, sizes=None):
        '''
        :param sizes: list represents network layers, a[i] - number of neurons in i-th layer
        '''
        self.weights = []
        self.inputs = []
        self.sizes = sizes

        if sizes is None:
            return

        self.layers_count = len(sizes)
        for i in range(self.layers_count - 1):
            theta_i = np.random.rand(sizes[i + 1], sizes[i])
            self.weights.append(theta_i.copy())

        self.inputs = [np.zeros((i,)) for i in sizes]

    def cost_func(self, y_true, y_pred):
        return sum((y_true - y_pred) ** 2)

    def train(self, x, y, learning_rate=0.2, eps=0.01):
        # TODO if layers is None set default arch
        x = np.array(x)
        n = len(x)
        epochs = 7000
        for epoch in range(epochs):
            mse = 0

            for i in range(n):
                y_pred = self.feed_forward(x[i])
                mse += self.cost_func(y[i], y_pred)

                deltas = [np.zeros((i,)) for i in self.sizes[1:]]
                delta_output = np.multiply(sigmoid_prime(y_pred), y[i] - y_pred)
                deltas[self.layers_count - 2] = delta_output

                for j in range(self.layers_count - 3, -1, -1):
                    deriv = sigmoid_prime(self.inputs[j + 1])
                    delta = np.multiply(deriv, np.dot(deltas[j + 1], self.weights[j + 1]))
                    deltas[j] = delta

                for j in range(self.layers_count - 1):
                    self.weights[j] += np.multiply(learning_rate, np.outer(deltas[j], self.inputs[j]))

            mse /= n
            if epoch % 1000 == 0:
                print("[{}] MSE: {}".format(epoch, mse))

    def feed_forward(self, x):

        self.inputs[0] = x
        for i in range(self.layers_count - 1):
            self.inputs[i + 1] = sigmoid(np.dot(self.weights[i], self.inputs[i]))
        return self.inputs[-1]

    def predict(self, x):
        return self.feed_forward(x)

    def save_weights(self, fname):
        np.save(fname, np.array(self.weights))

    def load_weights(self, fname):
        self.weights = np.load(fname)
