import numpy as np
import cv2 as cv
import math

class ThParameters:
    def __init__(self):
        self.levelNoise=6
        self.angle=45
        self.scale10=5
        self.origin=10
        self.xg=150
        self.yg=150
        self.update=True

def UpdateShape(x ):
    p.update = True

def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return () # or (0,0,0,0) ?
  return (x, y, w, h)

def NoisyPolygon(pRef,n):
#    vector<Point> c
    p = pRef;
#    vector<vector<Point> > contour;
    p = p+n*np.random.random_sample((p.shape[0],p.shape[1]))-n/2.0
    if (n==0):
        return p
    c = np.empty(shape=[0, 2])
    minX = p[0][0]
    maxX = p[0][0]
    minY = p[0][1]
    maxY = p[0][1]
    for i in range( 0,p.shape[0]):
        next = i + 1;
        if (next == p.shape[0]):
            next = 0;
        u = p[next] - p[i]
        d = int(cv.norm(u))
        a = np.arctan2(u[1], u[0])
        step = 1
        if (n != 0):
            step = d // n
        for j in range( 1,int(d),int(max(step, 1))):
            while  True:
                pAct = (u*j) / (d)
                r = n*np.random.random_sample()
                theta = a + 2*math.pi*np.random.random_sample()
#                pNew = Point(Point2d(r*cos(theta) + pAct.x + p[i].x, r*sin(theta) + pAct.y + p[i].y));
                pNew = np.array([(r*np.cos(theta) + pAct[0] + p[i][0], r*np.sin(theta) + pAct[1] + p[i][1])])
                if (pNew[0][0]>=0 and pNew[0][1]>=0):
                    break
            if (pNew[0][0]<minX):
                minX = pNew[0][0]
            if (pNew[0][0]>maxX):
                maxX = pNew[0][0]
            if (pNew[0][1]<minY):
                minY = pNew[0][1]
            if (pNew[0][1]>maxY):
                maxY = pNew[0][1]
            c = np.append(c,pNew,axis = 0)
    return c

#static vector<Point> NoisyPolygon(vector<Point> pRef, double n);
#static void UpdateShape(int , void *r);
#static void AddSlider(String sliderName, String windowName, int minSlider, int maxSlider, int valDefault, int *valSlider, void(*f)(int, void *), void *r);
def AddSlider(sliderName,windowName,minSlider,maxSlider,valDefault, update):
    cv.createTrackbar(sliderName, windowName, valDefault,maxSlider-minSlider+1, update)
    cv.setTrackbarMin(sliderName, windowName, minSlider)
    cv.setTrackbarMax(sliderName, windowName, maxSlider)
    cv.setTrackbarPos(sliderName, windowName, valDefault)

#    vector<Point> ctrRef;
#    vector<Point> ctrRotate, ctrNoisy, ctrNoisyRotate, ctrNoisyRotateShift;
#    // build a shape with 5 vertex
ctrRef = np.array([(250,250),(400, 250),(400, 300),(250, 300),(180, 270)])
cg = np.mean(ctrRef,axis=0)
p=ThParameters()
cv.namedWindow("FD Curve matching");
# A rotation with center at (150,150) of angle 45 degrees and a scaling of 5/10
AddSlider("Noise", "FD Curve matching", 0, 20, p.levelNoise,  UpdateShape)
AddSlider("Angle", "FD Curve matching", 0, 359, p.angle,  UpdateShape)
AddSlider("Scale", "FD Curve matching", 5, 100, p.scale10, UpdateShape)
AddSlider("Origin", "FD Curve matching", 0, 100, p.origin, UpdateShape)
AddSlider("Xg", "FD Curve matching", 150, 450, p.xg, UpdateShape)
AddSlider("Yg", "FD Curve matching", 150, 450, p.yg, UpdateShape)
code = 0
img = np.zeros((300,512,3), np.uint8)
print ("******************** PRESS g TO MATCH CURVES *************\n")

while (code!=27):
    code = cv.waitKey(60)
    if p.update:
        p.levelNoise=cv.getTrackbarPos('Noise','FD Curve matching')
        p.angle=cv.getTrackbarPos('Angle','FD Curve matching')
        p.scale10=cv.getTrackbarPos('Scale','FD Curve matching')
        p.origin=cv.getTrackbarPos('Origin','FD Curve matching')
        p.xg=cv.getTrackbarPos('Xg','FD Curve matching')
        p.yg=cv.getTrackbarPos('Yg','FD Curve matching')

        r = cv.getRotationMatrix2D((p.xg, p.yg), angle=p.angle, scale=10.0/ p.scale10);
        ctrNoisy= NoisyPolygon(ctrRef,p.levelNoise)
        ctrNoisy1 = np.reshape(ctrNoisy,(ctrNoisy.shape[0],1,2))
        ctrNoisyRotate = cv.transform(ctrNoisy1,r)
        ctrNoisyRotateShift = np.empty([ctrNoisyRotate.shape[0],1,2],dtype=np.int32)
        for  i in range(0,ctrNoisy.shape[0]):
            k=(i+(p.origin*ctrNoisy.shape[0])//100)% ctrNoisyRotate.shape[0]
            ctrNoisyRotateShift[i] = ctrNoisyRotate[k]
#       To draw contour using drawcontours
        cc= np.reshape(ctrNoisyRotateShift,[ctrNoisyRotateShift.shape[0],2])
        c = [ ctrRef,cc]
        p.update = False;
        rglobal =(0,0,0,0)
        for i in range(0,2):
            r = cv.boundingRect(c[i])
            rglobal = union(rglobal,r)
        r = list(rglobal)
        r[2] = r[2]+10
        r[3] = r[3]+10
        rglobal = tuple(r)
        img = np.zeros((2 * rglobal[3], 2 * rglobal[2], 3), np.uint8)
        cv.drawContours(img, c, 0, (255,0,0),1);
        cv.drawContours(img, c, 1, (0, 255, 0),1);
        cv.circle(img, tuple(c[0][0]), 5, (255, 0, 0),3);
        cv.circle(img, tuple(c[1][0]), 5, (0, 255, 0),3);
        cv.imshow("FD Curve matching", img);
    if code == ord('d') :
        cv.destroyWindow("FD Curve matching");
        cv.namedWindow("FD Curve matching");
# A rotation with center at (150,150) of angle 45 degrees and a scaling of 5/10
        AddSlider("Noise", "FD Curve matching", 0, 20, p.levelNoise,  UpdateShape)
        AddSlider("Angle", "FD Curve matching", 0, 359, p.angle,  UpdateShape)
        AddSlider("Scale", "FD Curve matching", 5, 100, p.scale10,  UpdateShape)
        AddSlider("Origin%%", "FD Curve matching", 0, 100, p.origin, UpdateShape)
        AddSlider("Xg", "FD Curve matching", 150, 450, p.xg,  UpdateShape)
        AddSlider("Yg", "FD Curve matching", 150, 450, p.yg,  UpdateShape)
    if  code == ord('g'):
        fit = cv.ximgproc.createContourFitting(1024,16);
# sampling contour we want 256 points
        cn= np.reshape(ctrRef,[ctrRef.shape[0],1,2])

        ctrRef2d = cv.ximgproc.contourSampling(cn,  256)
        ctrRot2d = cv.ximgproc.contourSampling(ctrNoisyRotateShift,  256)
        fit.setFDSize(16)
        c1 = ctrRef2d
        c2 = ctrRot2d
        alphaPhiST, dist	 = fit.estimateTransformation(ctrRot2d, ctrRef2d)
        print( "Transform *********\n Origin = ", 1-alphaPhiST[0,0] ," expected ", p.origin / 100. ,"\n")
        print( "Angle = ", alphaPhiST[0,1] * 180 / math.pi ," expected " , p.angle,"\n")
        print( "Scale = " ,alphaPhiST[0,2] ," expected " , p.scale10 / 10.0 , "\n")
        dst = cv.ximgproc.transformFD(ctrRot2d, alphaPhiST,cn, False);
        ctmp= np.reshape(dst,[dst.shape[0],2])
        cdst=ctmp.astype(int)

        c = [ ctrRef,cc,cdst]
        cv.drawContours(img, c, 2, (0,0,255),1);
        cv.circle(img, (int(c[2][0][0]),int(c[2][0][1])), 5, (0, 0, 255),5);
        cv.imshow("FD Curve matching", img);
