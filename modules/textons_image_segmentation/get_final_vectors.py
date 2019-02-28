import cv2
import sys
import numpy as np

class createVector(object):
    """
    This class creates the final vector
    """

    def __init__(self, vector, image):
        self.v = vector
        self.im = image
    def generateVectors(self):
        """
        generate feature vector with row, col and cluster center values
        """
        hold = []
        for i in range(self.im.shape[0]):
            for j in range(self.im.shape[1]):
                hold.append([i, j, -1])
        hold = np.array(hold)
        complete_vector = np.concatenate((self.v, hold), axis= 1)
        #print("complete vector ...")
        #print(complete_vector.shape)
        return complete_vector