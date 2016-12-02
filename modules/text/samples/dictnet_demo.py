#!/usr/bin/env python

import cv2
import sys
import os.path
 
#Global variable shared between the Mouse callback and main
refPt = []
cropping = False
image=None
drawImage=None
dictNet=None


def mouseCallback(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping,dictNet,drawImage,image

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        roi = image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        res=dictNet.classify([roi])
        drawImage = image.copy()
        cv2.rectangle(drawImage, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.putText(drawImage,"%s:%f"%(res[0][0],res[1][0]),refPt[0],cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.imshow("Select A Region", drawImage)
        

if __name__=='__main__':
    helpStr="""    Usage: """+sys.argv[0]+""" IMAGE_FILENAME
    
    Press 'q' or 'Q' exit
    
    The modelFiles must be available in the current directory.
    In linux shell they can be downloaded (~2GB) with the following commands:
    wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg.caffemodel
    wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_deploy.prototxt
    wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_labels.txt
    """
    if((len(sys.argv)!=2 )or not(os.path.isfile(sys.argv[1]) )):
        print helpStr
        print 'No image file given Aborting!'
        sys.exit(1)
    if not (os.path.isfile('dictnet_vgg_deploy.prototxt') and 
        os.path.isfile('dictnet_vgg.caffemodel') and
        os.path.isfile('dictnet_vgg_labels.txt')):
        print helpStr
        print 'Model files not present, Aborting!'
        sys.exit(1)

    dictNet=cv2.text.OCRDictnet_create('dictnet_vgg_deploy.prototxt',
                                       'dictnet_vgg.caffemodel',
                                       'dictnet_vgg_labels.txt','',1,0,1)
    image = cv2.imread(sys.argv[1])
    drawImage = image.copy()
    cv2.namedWindow("Select A Region")
    cv2.setMouseCallback("Select A Region", mouseCallback)

    while True:
        cv2.imshow("Select A Region", drawImage)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, break from the loop
        if key == ord("q") or key == ord("Q"):
            break

    cv2.destroyAllWindows()