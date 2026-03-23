import numpy as np
import matplotlib.pyplot as plt

def generateData(clusters, minPoints, maxPoints):
    Clus = []    
    Lbls = []

    for i in range(clusters):
        Clus.append(np.random.randn(np.random.randint(minPoints, maxPoints), 2) + np.random.uniform(0, 3, size=(1, 2)))
        Lbls.extend([i] * Clus[i].shape[0])

    X = np.vstack(Clus)
    Lbls = np.array(Lbls)

    plt.scatter(X[:, 0], X[:, 1], c=Lbls, cmap='viridis', alpha=0.7)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Generated Cluster Data")
    plt.show()

    return X, Lbls

def initializeCentroids(X, clusters):
    initialCentrois = X[np.random.choice(X.shape[0], clusters, replace = False)]

def assignClusters(X, centroids):
    dis = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    clusterLabels = np.argmin(dis, axis=1)
    return clusterLabels

def updateCentroids(X, clusterLabels, centroids):
    Cntrds = np.zeros((centroids, X.shape[1]))
    for i in range(centroids):
        Pic = X[clusterLabels == i]
        Cntrds[i] = Pic.mean(axis=0)

    return Cntrds

if __name__ == "__main__":
    X, Lbls = generateData(3, 50, 55)