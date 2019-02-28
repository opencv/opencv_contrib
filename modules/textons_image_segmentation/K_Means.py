import cv2
import sys
import numpy as np
import warnings
warnings.filterwarnings('error')

class KMeansLMfilters(object):
    """
    apply kmeans on the feature vector extracted

    output:
        1. updated numpy array for the image
        2. centroid values
    """
    def __init__(self, X, no_of_clusters, no_of_iterations):
        self.X = X
        self.clusters = int(no_of_clusters)
        self.iterations = int(no_of_iterations)
        
    def initializ_centroids(self):
        """
        randomly select data points as centers
        """
        centers =  self.X[np.random.choice(self.X.shape[0], self.clusters, replace=False), :]
        return centers[:,:self.X.shape[1]-3]

    def EuclideanDistance(self, X, Y):
        """
        return euclidean distace between two vectors 
        """
        return np.sqrt(np.sum((X-Y)**2, axis=1))

    def assign_centers(self, centroids):
        """
        assignn closest center to the vecctor
        """
        for features in range(self.X.shape[0]):
            mindist = self.EuclideanDistance(centroids,self.X[features][:self.X.shape[1]-3])
            index = np.argmin(mindist)
            self.X[features][self.X.shape[1]-1] = index
        return self.X

    def update_centroids(self):
        """
        update cluster centers
        """
        class_dict = {}
        for _ in range(self.clusters):
            class_dict[_] = list(self.X[i,:self.X.shape[1]-3] for i in np.asarray((self.X[:,self.X.shape[1]-1]==_).nonzero()))
        for _ in range(self.clusters):
            temp = np.array(class_dict[_])
            class_dict[_] = list(np.mean(np.reshape(temp, (temp.shape[1],temp.shape[2])),axis=0))
        new_centroids = []
        for _ in range(self.clusters):
            new_centroids.append(class_dict[_])
        centers = np.array(new_centroids)
        return centers

    def kMeans(self):
        """
        apply kMeans
        """
        rerun_flag = 0
        while(rerun_flag==0):
            centers = self.initializ_centroids()
            try:
                for _ in range(self.iterations):
                    print("[",end = "" )
                    print("#"*(_+1), end="")
                    print("_"*(self.iterations-_-1), end="")
                    print("]", end=" ")
                    print("iterations completed:" + str(_+1)+"/" + str(self.iterations))
                    self.X = self.assign_centers(centers)
                    centers = self.update_centroids()
                rerun_flag = -1
            except Warning:
                print("Rerun!! Problem with centroid alloaction!")
                print("Try changing the values of number of clusters and/or number of iterations!")
            
        return self.assign_centers(centers)