import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


learningRate_ = 0.015
epochs = 2000

def relu(x):
    return np.maximum(0, x)

def reluDerivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid (used for backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)

class NN:
    def __init__(self, inSize,  outSize, hiddenLayers):
        self.inSize = inSize
        self.hiddenLayers = hiddenLayers
        self.outSize = outSize

        layerSizes = [inSize] + hiddenLayers + [outSize] 
        
        self.weights = []
        self.biases = []

        for i in range(len(layerSizes) - 1): 
            w = np.random.randn(layerSizes[i], layerSizes[i+1]) * .1
            b = np.random.rand(1, layerSizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        activations = [X]
        a = X

        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = relu(z)
            activations.append(z)

        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        output = sigmoid(z)  # Sigmoid for output layer
        activations.append(output)
        
        return activations

    def backward(self, X, activations, y, learningRate):
        deltas = []

        error = y - activations[-1]
        delta = error * sigmoid_derivative(activations[-1])
        deltas.append(delta)

        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1].dot(self.weights[i+1].T) * reluDerivative(activations[i+1])
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            n = X.shape[0]
            self.weights[i] += activations[i].T.dot(deltas[i] / n) * learningRate
            self.biases[i] += (np.sum(deltas[i], axis=0, keepdims=True) / n) * learningRate

    def fit(self, X, y, learningRate, epochs):
        for epoch in range(epochs):
            #Forward pass to get outputs
            activations = self.forward(X)

            #Backward pass to apply back propogation
            self.backward(X, activations, y, learningRate)

            if epoch % 100 == 0:
                loss = np.mean(np.square(y - activations[-1]))
                print(f"Loss on epoch: {epoch}: {loss}\n")

    def predict(self, X):
        return self.forward(X)[-1]

def generateData():
    iris = load_iris()
    X, y = iris.data, iris.target

    X = X[y != 2]
    y = y[y != 2]

    y = y.reshape(-1, 1)

    Xr, Xt, Yr, Yt = train_test_split(X, y, test_size=0.3)


    return Xr, Xt, Yr, Yt

if __name__ == "__main__":
    Xr, Xt, Yr, Yt = generateData()
    
    nn = NN(inSize=4, hiddenLayers=[5, 5], outSize=1)
    nn.fit(Xr, Yr, epochs=epochs, learningRate=learningRate_)

    preds = nn.predict(Xr)
    pred_labels = (preds > 0.5).astype(int)

    print("Predictions (last 10):")
    print(pred_labels[-10:])

    print("Actual (last 10):")
    print(Yt[-10:])

    accuracy = np.mean(pred_labels == Yr)
    print(f"Accuracy: {accuracy * 100:.2f}%")

        






        

