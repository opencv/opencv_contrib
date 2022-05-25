import sys
import numpy as np
import cv2 as cv

def AddSlider(sliderName,windowName,minSlider,maxSlider,valDefault, update=[]):
    if update is None:
        cv.createTrackbar(sliderName, windowName, valDefault,maxSlider-minSlider+1)
    else:
        cv.createTrackbar(sliderName, windowName, valDefault,maxSlider-minSlider+1, update)
    cv.setTrackbarMin(sliderName, windowName, minSlider)
    cv.setTrackbarMax(sliderName, windowName, maxSlider)
    cv.setTrackbarPos(sliderName, windowName, valDefault)
class Filtrage:
    def __init__(self):
        self.s =0
        self.alpha = 100
        self.omega = 100
        self.updateFiltre=True
        self.img=[]
        self.dximg=[]
        self.dyimg=[]
        self.module=[]
    def DericheFilter(self):
        self.dximg = cv.ximgproc.GradientDericheX(	self.img, self.alpha/100., self.omega/1000.	)
        self.dyimg = cv.ximgproc.GradientDericheY(	self.img, self.alpha/100., self.omega/1000.	)
        dx2=self.dximg*self.dximg
        dy2=self.dyimg*self.dyimg
        self.module = np.sqrt(dx2+dy2)
        cv.normalize(src=self.module,dst=self.module,norm_type=cv.NORM_MINMAX)
    def SlideBarDeriche(self):
        cv.namedWindow(self.filename)
        AddSlider("alpha",self.filename,1,400,self.alpha,self.UpdateAlpha)
        AddSlider("omega",self.filename,1,1000,self.omega,self.UpdateOmega)

    def UpdateOmega(self,x ):
        self.updateFiltre=True
        self.omega=x
    def UpdateAlpha(self,x ):
        self.updateFiltre=True
        self.alpha=x
    def run(self,argv):
        # Load the source image
        self.filename = argv[0] if len(argv) > 0 else "../doc/pics/corridor_fld.jpg"
        self.img=cv.imread(self.filename,cv.IMREAD_GRAYSCALE)
        if self.img is None:
            print ('cannot read file')
            return
        self.SlideBarDeriche()
        while True:
            cv.imshow(self.filename,self.img)
            if self.updateFiltre:
                self.DericheFilter()
                cv.imshow("module",self.module)
                self.updateFiltre =False
            code = cv.waitKey(10)
            if code==27:
                break
if __name__ == '__main__':
    Filtrage().run(sys.argv[1:])
