import numpy as np

learningRate_ = 0.1

def sigmoid(x):
    return 1/(1+np.exp(-x))

#For backProp
def sigmoidDerivative(x):
    return x * (1 - x)

class NN:
    def __init__(self, layerSizes):
        self.layerSizes = layerSizes
        self.weights = []
        self.biases = []

        for i in range(len(layerSizes) - 1): 
            w = np.random.randn(layerSizes[i], layerSizes[i+1])
            b = np.random.rand(1, layerSizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        activations = [X]
        a = X

        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = sigmoid(z)
            activations.append(a)
        
        return activations

    def backward(self, activations, y):
        deltas = []

        error = y - activations[-1]
        delta = error * sigmoidDerivative(activations[-1])
        deltas.append(delta)

        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1].dot(self.weights[i+1].T) * sigmoidDerivative(activations[i])
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] += activations[i].T.dot(deltas[i]) * learningRate_
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learningRate_

        

