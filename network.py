import numpy as np

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
            self.layers_count = 0
            return

        self.layers_count = len(sizes)
        self.init_weights()

    def init_weights(self):
        self.layers_count = len(self.sizes)
        for i in range(self.layers_count - 1):
            theta_i = np.random.rand(self.sizes[i + 1], self.sizes[i])
            self.weights.append(theta_i.copy())

        self.inputs = [np.zeros((i,)) for i in self.sizes]

    def cost_func(self, y_true, y_pred):
        return sum((y_true - y_pred) ** 2)

    def train(self, x, y, learning_rate=0.2, epochs=100, verbose=10):

        if self.sizes is None:
            self.sizes = [x.shape[1], x.shape[1], y.shape[1]]
            self.init_weights()

        n = len(x)
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
            if epoch % verbose == 0:
                print("[{}] MSE: {}".format(epoch, mse))

    def feed_forward(self, x):

        self.inputs[0] = x
        for i in range(self.layers_count - 1):
            self.inputs[i + 1] = sigmoid(np.dot(self.weights[i], self.inputs[i]))
        return self.inputs[-1]

    def predict(self, x):
        res = []
        for x_i in x:
            res.append(self.feed_forward(x_i))
        return res

    def save_weights(self, fname):
        np.save(fname, np.array(self.weights))

    def load_weights(self, fname):
        self.weights = np.load(fname)
