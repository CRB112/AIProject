import numpy as np
import matplotlib.pyplot as plt

def generateData(numClus, minPoints, maxPoints):
    #Clusters and labels will be randomly generated
    Clus = []    
    Lbls = []

    #Creates numClus new clusters with minPoints to maxPoints values
    for i in range(numClus):
        Clus.append(np.random.randn(np.random.randint(minPoints, maxPoints), 2) + np.random.uniform(0, 3, size=(1, 2)))
        #Lbls is a 1d array of points associated with clusters
        #If a cluster has 50 points, Lbls[0-49] are going to be cluster 0
        #Not convenient for readability but makes coding much simpler
        Lbls.extend([i] * Clus[i].shape[0])

    #Stacks all of the clusters into ONE array
    X = np.vstack(Clus)
    Lbls = np.array(Lbls)

    #Showing data
    plt.scatter(X[:, 0], X[:, 1], c=Lbls, cmap='viridis', alpha=0.7)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Generated Cluster Data")
    plt.show()

    return X, Lbls

#Picks numClus random points from X
def initializeCentroids(X, numClus):
    initialCentroids = X[np.random.choice(X.shape[0], numClus, replace = False)]
    return initialCentroids

#Assigns points to clusters
#Uses np.linalg.norm to assign distances to each cluster according
#To point
#Adds the cluster label of the closest cluster found
def assignClusters(X, centroids):
    dis = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    clusterLabels = np.argmin(dis, axis=1)
    return clusterLabels

#Moves centroids to the average of the distances
#From each of the points
def updateCentroids(X, clusterLabels, numClus):
    Cntrds = np.zeros((numClus, X.shape[1]))
    for i in range(numClus):
        Pic = X[clusterLabels == i]
        if len(Pic) > 0:
            Cntrds[i] = Pic.mean(axis=0)
    return Cntrds

#Uses tolerance to decide when to end
#MAXITER is used for safety from infinite looping
#Runs assignClusters and updateCentroids repeatedly until
#One of these contitions is met
def kMeans(X, numClus, MAXITER=1000, TOLERANCE=.00001):
    centroids = initializeCentroids(X, numClus)
    for _ in range(MAXITER):
        lbls = assignClusters(X, centroids)
        Cntrds = updateCentroids(X, lbls, numClus)

        shift = np.linalg.norm(centroids - Cntrds)
        if shift < TOLERANCE:
            break
            
        centroids = Cntrds

    return centroids, lbls


if __name__ == "__main__":
    numClus = int(input("Input cluster amount ->"))
    X, Lbls = generateData(numClus, 50, 55)
    centroids, labels = kMeans(X, numClus)

    #Data visualization
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
    plt.title("K-Means Result")
    plt.show()