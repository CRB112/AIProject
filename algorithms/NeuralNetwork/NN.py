import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

#Using iris variables
#Loaded matplot graph will only show two of the four
#But will satisfy in visualizing spread



#Relu is better for hidden layers since it dynamically returns
#Higher values from nodes with more effect
def relu(x):
    return np.maximum(0, x)

#Derivative for relu, used in back propogation
def reluDerivative(x):
    return (x > 0).astype(float)

#Sigmoid is used for final output, since output is going to be binary 1, 0
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid (used for backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)

#Main neural network class
class NN:
    #Runs at construction 
    #Paramaterized layers
    def __init__(self, inSize,  outSize, hiddenLayers):
        self.inSize = inSize
        self.hiddenLayers = hiddenLayers
        self.outSize = outSize

        #layerSizes is a 2d array of dynamic column sizes
        layerSizes = [inSize] + hiddenLayers + [outSize] 
        
        #Initial weights and biases of layers
        self.weights = []
        self.biases = []

        #Assigning random initial weights and biases to the layers
        for i in range(len(layerSizes) - 1): 
            w = np.random.randn(layerSizes[i], layerSizes[i+1]) * .1
            b = np.random.rand(1, layerSizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

    #Forward passes, saves activations in activations for back propogation
    def forward(self, X):
        activations = [X]
        a = X

        #For each layer, compute the dot product with the past biases and current weights
        #the activations of the layer are computied using relu()
        #Which is 0 if activation is negative and scales with positivity
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = relu(z)
            activations.append(a)

        #Final layer, seperate because sigmoid is used for binary decisions in which flower
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        output = sigmoid(z)  # Sigmoid for output layer
        activations.append(output)
        
        return activations

    #Back propogation
    def backward(self, X, activations, y, learningRate):
        #Deltas is the accumulation of layers and their respective activation biases
        deltas = []

        #Initial error on the last layer, needed for the loop
        #To run properly
        error = y - activations[-1]
        delta = error * sigmoid_derivative(activations[-1])
        deltas.append(delta)

        #Iterates backwards through the activations and obtains the delta of the layer
        #aka the 'error' according the the reluDerivative()
        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1].dot(self.weights[i+1].T) * reluDerivative(activations[i+1])
            deltas.append(delta)

        #Reverses the dataset since deltas was created backwards
        deltas.reverse()

        #Modifies the weights and biases of layers according to the deltas obtained
        for i in range(len(self.weights)):
            n = X.shape[0]
            self.weights[i] += activations[i].T.dot(deltas[i] / n) * learningRate
            self.biases[i] += (np.sum(deltas[i], axis=0, keepdims=True) / n) * learningRate

    #Runs the NeuralNetwork forwards and backwards
    def fit(self, X, y, learningRate, epochs):
        for epoch in range(epochs):
            #Forward pass to get outputs
            activations = self.forward(X)

            #Backward pass to apply back propogation
            self.backward(X, activations, y, learningRate)

            #Prints current loss after every 100 epochs
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - activations[-1]))
                print(f"Loss on epoch: {epoch}: {loss}\n")

    #Gets the expected output
    def predict(self, X):
        return self.forward(X)[-1]

#Grabs data from the iris dataset
def generateData():
    iris = load_iris()
    
    #X is input, y is output
    X, y = iris.data, iris.target

    #Trims the sets down to two flower types instead of 3
    #Really only for ease of programming and allows
    #The use of sigmoid for final output
    X = X[y != 2]
    y = y[y != 2]

    #Turns the output into a 1D matrix instead of an array
    #For dot product
    y = y.reshape(-1, 1)

    #Splitting data into training and testing matrices, %70/30
    Xr, Xt, Yr, Yt = train_test_split(X, y, test_size=0.3)



    #-----------------------------------------------
    #From here down is just for data visualization And Input
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['species'] = [iris.target_names[i] for i in y.flatten()]

    # Scatter plot: Sepal Length vs Sepal Width
    plt.figure(figsize=(8, 6))
    species_colors = {'setosa': 'blue', 'versicolor': 'green'}

    for species, color in species_colors.items():
        subset = df[df['species'] == species]
        plt.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],
                    label=species, color=color, alpha=0.7)

    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('Sepal Length vs Sepal Width (2 Species)')
    plt.legend()
    plt.show()

    epochs = int(input("How many Epochs would you like? ->"))
    learningRate = float(input("What learning rate would you like? ->"))
    hiddenLayerCount = int(input("How many hidden layers would you like? ->"))
    layers = []
    for i in range(hiddenLayerCount):
        layers.append(int(input(f"Input node count for layer {i}")))


    return Xr, Xt, Yr, Yt, layers, epochs, learningRate

if __name__ == "__main__":
    Xr, Xt, Yr, Yt, layers, epochs, learningRate = generateData()
    
    nn = NN(inSize=4, hiddenLayers=layers, outSize=1)
    nn.fit(Xr, Yr, epochs=epochs, learningRate=learningRate)

    preds = nn.predict(Xt)
    pred_labels = (preds > 0.5).astype(int)

    pred_species = ['setosa' if x==0 else 'versicolor' for x in pred_labels.flatten()]
    actual_species = ['setosa' if x==0 else 'versicolor' for x in Yt.flatten()]

    print("Predictions (last 10):")
    print(pred_species[-10:])

    print("Actual (last 10):")
    print(actual_species[-10:])

    accuracy = np.mean(pred_labels == Yt)
    print(f"Accuracy: {accuracy * 100:.2f}%")

        






        

