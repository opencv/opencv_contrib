import cv2
import sys
import numpy as np

class LMFilters(object):
    """
    generate LM filters for textons
    """
    def __init__(self):
        pass
    
    def gaussian1d(self, sigma, mean, x, ord):
        """
        return gaussian differentiation for 1st and 2nd order deravatives
        """
        x = np.array(x)
        x_ = x - mean
        var = sigma**2
        # Gaussian Function
        g1 = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x_*x_)/(2*var)))

        if ord == 0:
            g = g1
            return g #gaussian function
        elif ord == 1:
            g = -g1*((x_)/(var))
            return g #1st order differentiation
        else:
            g = g1*(((x_*x_) - var)/(var**2))
            return g #2nd order differentiation

    def gaussian2d(self,sup, scales):
        var = scales * scales
        shape = (sup,sup)
        n,m = [(i - 1)/2 for i in shape]
        x,y = np.ogrid[-m:m+1,-n:n+1]
        g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
        return g

    def log2d(self,sup, scales):
        var = scales * scales
        shape = (sup,sup)
        n,m = [(i - 1)/2 for i in shape]
        x,y = np.ogrid[-m:m+1,-n:n+1]
        g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
        h = g*((x*x + y*y) - var)/(var**2)
        return h

    def makefilter(self, scale, phasex, phasey, pts, sup):

        gx = self.gaussian1d(3*scale, 0, pts[0,...], phasex)
        gy = self.gaussian1d(scale,   0, pts[1,...], phasey)

        image = gx*gy

        image = np.reshape(image,(sup,sup))
        return image

    def makeLMfilters(self):
        sup     = 49
        scalex  = np.sqrt(2) * np.array([1,2,3])
        norient = 6
        nrotinv = 12

        nbar  = len(scalex)*norient
        nedge = len(scalex)*norient
        nf    = nbar+nedge+nrotinv
        F     = np.zeros([sup,sup,nf])
        hsup  = (sup - 1)/2

        x = [np.arange(-hsup,hsup+1)]
        y = [np.arange(-hsup,hsup+1)]

        [x,y] = np.meshgrid(x,y)

        orgpts = [x.flatten(), y.flatten()]
        orgpts = np.array(orgpts)

        count = 0
        for scale in range(len(scalex)):
            for orient in range(norient):
                angle = (np.pi * orient)/norient
                c = np.cos(angle)
                s = np.sin(angle)
                rotpts = [[c+0,-s+0],[s+0,c+0]]
                rotpts = np.array(rotpts)
                rotpts = np.dot(rotpts,orgpts)
                F[:,:,count] = self.makefilter(scalex[scale], 0, 1, rotpts, sup)
                F[:,:,count+nedge] = self.makefilter(scalex[scale], 0, 2, rotpts, sup)
                count = count + 1

        count = nbar+nedge
        scales = np.sqrt(2) * np.array([1,2,3,4])

        for i in range(len(scales)):
            F[:,:,count]   = self.gaussian2d(sup, scales[i])
            count = count + 1

        for i in range(len(scales)):
            F[:,:,count] = self.log2d(sup, scales[i])
            count = count + 1

        for i in range(len(scales)):
            F[:,:,count] = self.log2d(sup, 3*scales[i])
            count = count + 1

        return F