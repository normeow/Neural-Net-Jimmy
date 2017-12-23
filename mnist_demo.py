from network import NeuralNetwork
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn import datasets

digits = datasets.load_digits()
X = preprocessing.scale(digits.data.astype(float))
encoder = preprocessing.OneHotEncoder(sparse=False)
y = encoder.fit_transform(np.array(digits.target).reshape(-1, 1))

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

print("Training...")
jimmy = NeuralNetwork()
jimmy.train(X_train, y_train, verbose=10, epochs=50)
print("Done training")

jimmy.save_weights("mnist_weights.npy")
pred = jimmy.predict(X_valid)

pred = np.argmax(pred, axis=1)
y_valid = np.argmax(y_valid, axis=1)

print("Accuracy: ", metrics.accuracy_score(pred, y_valid))
