import cv2
import sys
import numpy as np

class preprocessImageWithKernles(object):
    """
    create vector features after applying kernels
    """
    def __init__(self, image):
        #convert image to grayscale
        self.im = image                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

    def merge(self):
        """
        apply gaussian blur
        """
        self.out = cv2.GaussianBlur(self.im,(5,5),0)
        

    def apply_kernel(self, kernels):
        """
        apply LM filter kernels on the image after preprocessing
        """
        image = []
        
        for i in range(kernels.shape[2]):
            image.append(cv2.filter2D(self.out, -1, kernels[:,:,i]))

        self.image_after_filters = np.array(image)
        #print("creating filtered image...")
        #print(self.image_after_filters.shape)
        
    def create_vectors(self):
        """
        return vectors created for each pixel
        """
        point_vector = list()
        for row in range(self.image_after_filters.shape[1]):
            for col in range(self.image_after_filters.shape[2]):
                point_vector.append(self.image_after_filters[:, row, col])
        self.pixel_point_vector = np.array(point_vector)
        #print(self.pixel_point_vector.shape)
        #print("initial vector...")
        return self.pixel_point_vector