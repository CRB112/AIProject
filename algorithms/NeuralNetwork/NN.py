import numpy as np

learningRate_ = 0.1
epochs = 5000

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

        #First (last) activations, the loop wont work
        #Without an initial value in deltas
        error = y - activations[-1]
        delta = error * sigmoidDerivative(activations[-1])
        deltas.append(delta)

        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1].dot(self.weights[i+1].T) * sigmoidDerivative(activations[i+1])
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            n = activations[i].shape[0]
            
            self.weights[i] += activations[i].T.dot(deltas[i] / n) * learningRate_
            self.biases[i] += (np.sum(deltas[i], axis=0, keepdims=True) / n) * learningRate_

    def fit(self, X, y):
        for epoch in range(epochs):
            
            #Forward pass to get outputs
            activations = self.forward(X)

            #Backward pass to apply back propogation
            self.backward(activations, y)

            if epoch % 100 == 0:
                loss = np.mean(np.square(y - activations[-1]))
                print(f"Loss on epoch: {epoch}: {loss}\n")

    def predict(self, X):
        return self.forward(X)[-1]

def generateData():
    X = np.random.rand(100, 2)

    y = (np.sum(X, axis=1) > 1).astype(int)
    y = y.reshape(-1, 1)

    return X, y

if __name__ == "__main__":
    X, y = generateData()
    
    nn = NN([2, 3, 5, 6, 1])
    nn.fit(X, y)

    preds = nn.predict(X)
    pred_labels = (preds > 0.5).astype(int)

    print("Predictions (first 10):")
    print(pred_labels[:10])

    print("Actual (first 10):")
    print(y[:10])

    accuracy = np.mean(pred_labels == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")

        






        

