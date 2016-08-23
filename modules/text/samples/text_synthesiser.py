#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import sys
import os.path
import time
from sys import platform
from commands import getoutput as go
import numpy as np


enLexicon=['hello','world','these','samples','they','can','be',
         'english','or','other','scripts']

grLexicon=['Ελληνικές','λέξεις','μπορούν','να','συντεθούν','εξίσου','εύκολα',
         'με','μόνη','διαφορά','την','έλλειψη','γραμματοσειρών']

colorClusters=(np.random.rand(12,3,3)*255).astype('uint8')


helpStr="""Usage: """+sys.argv[0]+""" scenetext_segmented_word01.jpg scenetext_segmented_word02.jpg ...

    This program is a demonstration generates 100 synthetic  images and displays 9 of them randomly selected.
    It acts as a benchmark as weel, providing an estimate on the resources need for the millions of samples needed
    for training word-spotting deep CNNs. The files you provide are used to crop natural patches for background
    """


if __name__=='__main__':
    if len(sys.argv)<2:
        print helpStr
        sys.exit()
    synthEng=cv2.text.TextSynthesizer_create(cv2.text.CV_TEXT_SYNTHESIZER_SCRIPT_LATIN)
    synthGr=cv2.text.TextSynthesizer_create(cv2.text.CV_TEXT_SYNTHESIZER_SCRIPT_GREEK)
    #we make Greek text more distorted with more curves
    synthGr.setCurvingProbabillity(.60)
    synthEng.setCurvingProbabillity(.05)

    for bgFname in sys.argv[1:]:
        img=cv2.imread(bgFname,cv2.IMREAD_COLOR)
        synthEng.addBgSampleImage(img)
        synthGr.addBgSampleImage(img)

    synthEng.setColorClusters(colorClusters)
    synthGr.setColorClusters(colorClusters)
    enCaptions=(enLexicon*20)[:100]
    grCaptions=(grLexicon*20)[:100]
    for k in range(9):
        cv2.namedWindow('En%d'%(k+1))
        cv2.moveWindow('En%d'%(k+1),30+(k%3)*500,30+(k/3)*160)
        cv2.namedWindow('Gr%d'%(k+1))
        cv2.moveWindow('Gr%d'%(k+1),30+(k%3)*500,530+(k/3)*160)
    t=time.time()
    #The actual sample generation is synthEng.generateSample(c)
    englishSamples=[synthEng.generateSample(c) for c in enCaptions]
    greekSamples=[synthGr.generateSample(c) for c in grCaptions]
    dur=time.time()-t
    print 'Generated 200 samples in ',int(dur*1000),' msec.\n'
    for k in range(9):
        cv2.imshow('En%d'%(k+1),englishSamples[k])
        cv2.imshow('Gr%d'%(k+1),greekSamples[k])

    print '\n\nPress (Q) to continue'
    k=cv2.waitKey();
    while k!=ord('Q') and k!=ord('q'):
        k=cv2.waitKey();
    cv2.destroyAllWindows()
