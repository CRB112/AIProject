import numpy as np
import matplotlib.pyplot as plt

#Can read data from a file in the form
#x,y
#x,y
#x,y
#If no file is found with filename, creates its own
#With paramaterized inputs and random noise
def readFile(filename):
    x=[]
    y=[]
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                xP, yP = (line.split(','))
                x.append(float(xP))
                y.append(float(yP))
    except FileNotFoundError:
        MAXPOINTS = int(input("How many random points would you like? ->"))
        m = np.random.uniform(-2, 2)
        b = np.random.uniform(-5, 5)

        with open(filename, 'a') as f:
            for _ in range(MAXPOINTS):
                x = np.random.uniform(0, 10)
                f.write(f"{x},{x*m + b + np.random.normal(0, 1)}\n")

        return readFile(filename)

    return np.array(x), np.array(y)

#Finds the means of x and the means of y
#To calculate the mean slope
def linearRegression(x, y):
    n = len(x)

    meanX = np.mean(x)
    meanY = np.mean(y)

    num = np.sum((x - meanX) * (y - meanY))
    den = np.sum((x - meanX) ** 2)
    m = num / den

    b = meanY - m * meanX

    return m, b

#y = mx + b
def predict(x, m, b):
    return m * x + b

#Runs the linearRegressions
if __name__ == "__main__":
    x, y, = readFile('input.txt')
    m, b = linearRegression(x, y)

    
    print(f"Slope (m): {m}")
    print(f"Intercept (b): {b}")

    # Prediction
    x_new = float(input("Enter X value: "))
    y_pred = predict(x_new, m, b)
    print(f"Predicted Y: {y_pred}")

    # Plot
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.plot(x, m * x + b, color='red', label='Best Fit Line')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()
