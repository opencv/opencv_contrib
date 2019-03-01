import cv2
import sys
import numpy as np

class reconstructImage(object):
    def __init__(self, feature_vector, image, number_of_centers, assignment_type='RANDOM'):
        '''
        assignment type 'RANDOM' is suited more for higher number of clusters 
        assignment type 'DEFINED' is better suited when number of clusters is less than 15
        however the assignemnt types do not impact the performance of the code in any significant way
        '''

        self.v = feature_vector
        self.im = image
        self.c = number_of_centers
        self.type = assignment_type
    def reconstruct(self):
        """
        assign pixel values from the feature vectors
        """
        if(self.type==0):
            colors = np.random.randint(256, size=(int(self.c), 1))

        elif (self.type==1):
            colors = list()
            for i in range(0, 256, int(256//int(self.c))):
                colors.append([i])
            colors = np.array(colors)
        else:
            print('color scheme unavailable !!')
            print('using colors scheme "RANDOM"')

        image = np.zeros_like(self.im)
        for _ in range(self.v.shape[0]):
            x,y,k = self.v[_,48], self.v[_,49], self.v[_,50]
            image[x,y] = colors[k]

        return image