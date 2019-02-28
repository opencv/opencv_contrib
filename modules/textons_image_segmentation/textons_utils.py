import cv2
import sys
import numpy as np
from create_LM_filters import LMFilters
from convolve_LM_filters import preprocessImageWithKernles
from get_final_vectors import createVector
from K_Means import KMeansLMfilters
from reconstruct_image import reconstructImage
import random
random.seed(128)

class Textons(object):
    def __init__(self, image, cluster_centers, iterations, type_of_assignment):
        """
        inputs:
            1.image ==> numpy array (grayscaled image)
            2.number of cluster centers ==> integer
            3.number of iterations ==> integers
            4.type of colors assignment in the final image ==> integer 0 for 'RANDOM' or 1 for'DEFINED'

        output:
            numpy array of image after k means wiht LM filters
        """
        self.im = image
        try:

            if (self.im.shape[2]==3):
                print("color image")
                print("convert image to grayscale! ")
                sys.exit()
        except IndexError:
            pass

        self.cluster_centers = cluster_centers
        self.iterations = iterations
        self.type_of_assignment = type_of_assignment

    def textons(self):
        #create LM filters
        LM_filters = LMFilters()
        filters = np.array(LM_filters.makeLMfilters())#import filters as numpy array
        #check filters
        #print("filters created...")
        #print(filters.shape)
        #genrate vectors applying kMeans
        I = preprocessImageWithKernles(self.im)
        I.merge()
        I.apply_kernel(filters) 
        ##I.apply_kernel(np.load("LMkernels.npy"))
        features_for_kmeans = I.create_vectors()
        ##np.save("vectors_for_kmeans.npy", features_for_kmeans)
        #print("creating first vector set..")
        #print(features_for_kmeans.shape)
        #create final vectors
        V = createVector(features_for_kmeans, self.im)
        final_feature_vectors = V.generateVectors()
        #print("creating final vector set...")
        #print(final_feature_vectors.shape)
        #np.save("final_feature_vectors.npy", final_feature_vectors)
        ##apply k means on higher dimension image
        K = KMeansLMfilters(final_feature_vectors, no_of_clusters=self.cluster_centers,no_of_iterations=self.iterations)
        final = K.kMeans()
        #print("done")
        #reconstrcut image after k means
        final_image = reconstructImage(final,image=self.im,number_of_centers=self.cluster_centers, 
                                        assignment_type= self.type_of_assignment)
        display_image = final_image.reconstruct()
        #print('final image array ...')
        return display_image