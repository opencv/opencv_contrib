#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np

# Global Variable definition

helpStr="""Usage: """+sys.argv[0]+""" scenetext_segmented_word01.jpg scenetext_segmented_word02.jpg ...

    This program is a demonstration of the text syntheciser and the effect some of its parameters have.
    The file parameters are optional and they are used to sample backgrounds for the sample synthesis.
    
    In order to quit press (Q) or (q) while the window is in focus.
    """

colorClusters=(np.random.rand(20,3,3)*255).astype('uint8')
colorClusters[0,:,:]=[[255,0,255],[0,255,0],[128,32,32]]
colorClusters[0,:,:]=[[255,255,255],[0,0,0],[128,128,128]]
colorClusters[0,:,:]=[[0,0,0],[255,255,255],[32,32,32]]

words=['opencv','ανοιχτόCV','открытыйcv','مفتوحcv','פָּתוּחַcv']
[w[0].upper()+w[1:-2]+w[-2:].upper() for w in words]
synthlist=[cv2.text.TextSynthesizer_create(50,400,k+2) for k in range(5)]
script=0
s=synthlist[script]
word=words[script]
pause=200

# GUI Callsback functions

def updatePerspective(x):
    global s
    s.setMaxPerspectiveDistortion(x)

def updateCompression(x):
    global s
    s.setCompressionNoiseProb(x/100.0)

def updateCurvProb(x):
    global s
    s.setCurvingProbabillity(x/100.0)

def updateCurvPerc(x):
    global s
    s.setMaxHeightDistortionPercentage(float(x))

def updateCurvArch(x):
    global s
    s.setMaxCurveArch(x/500.0)

def switchScripts(x):
    global s
    global word
    global synthlist
    global script
    script=x
    s=synthlist[script]
    word=words[script]
    updateTrackbars()

def updateTime(x):
    global pause
    pause=x

def initialiseSynthesizers():
    global synthlist
    global colorClusters
    filenames=sys.argv[1:]
    for fname in filenames:
        img=cv2.imread(fname,cv2.IMREAD_COLOR)
        for synth in synthlist:
            synth.addBgSampleImage(img)
            synth.setColorClusters(colorClusters)
            synth.setMaxPerspectiveDistortion(0)
            synth.setCompressionNoiseProb(0)
            synth.setCurvingProbabillity(0)

# Other functions

def initWindows():
    global script
    global s
    global word
    global pause
    cv2.namedWindow('Text Synthesizer Demo',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Text Synthesizer Demo',1600,900)
    cv2.moveWindow('Text Synthesizer Demo',100,100)
    cv2.createTrackbar('Perspective','Text Synthesizer Demo',int(s.getMaxPerspectiveDistortion()),49,updatePerspective)
    cv2.createTrackbar('Compression','Text Synthesizer Demo',int(s.getCompressionNoiseProb()*100),100,updateCompression)
    cv2.createTrackbar('Curve Prob.','Text Synthesizer Demo',int(s.getCurvingProbabillity()*100),100,updateCurvProb)
    cv2.createTrackbar('Curve %','Text Synthesizer Demo',int(s.getMaxHeightDistortionPercentage()),10,updateCurvPerc)
    cv2.createTrackbar('Curve rad.','Text Synthesizer Demo',int(s.getMaxCurveArch()*500),100,updateCurvArch)
    cv2.createTrackbar('Script','Text Synthesizer Demo',int(script),4,switchScripts)
    cv2.createTrackbar('Pause ms','Text Synthesizer Demo',int(pause),500,updateTime)

def updateTrackbars():
    global script
    global s
    global word
    global pause
    cv2.setTrackbarPos('Perspective','Text Synthesizer Demo',int(s.getMaxPerspectiveDistortion()))
    cv2.setTrackbarPos('Compression','Text Synthesizer Demo',int(s.getCompressionNoiseProb()*100))
    cv2.setTrackbarPos('Curve Prob.','Text Synthesizer Demo',int(s.getCurvingProbabillity()*100))
    cv2.setTrackbarPos('Curve %','Text Synthesizer Demo',int(s.getMaxHeightDistortionPercentage()))
    cv2.setTrackbarPos('Curve rad.','Text Synthesizer Demo',int(s.getMaxCurveArch()*500))
    cv2.setTrackbarPos('Script','Text Synthesizer Demo',int(script))
    cv2.setTrackbarPos('Pause ms','Text Synthesizer Demo',int(pause))

def guiLoop():
    global script
    global s
    global word
    global pause
    k=''
    while ord('q')!=k:
        if pause<500:
            cv2.imshow('Text Synthesizer Demo',s.generateSample(word))
        k=cv2.waitKey(pause+1)

# Main Programm

if __name__=='__main__':
    colorImg=cv2.imread('1000_color_clusters.png',cv2.IMREAD_COLOR)
    #1000_color_clusters.png has the 3 most dominant color clusters 
    #from the first 1000 samples of MSCOCO-text trainset
    if colorImg!=None:
        colorClusters=colorImg
    print helpStr
    initialiseSynthesizers()
    initWindows()
    updateTrackbars()
    guiLoop()