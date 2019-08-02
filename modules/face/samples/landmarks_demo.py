import random
import numpy as np
import cv2 as cv

frame1 = cv.imread(cv.samples.findFile('lena.jpg'))
if frame1 is None:
    print("image not found")
    exit()
frame = np.vstack((frame1,frame1))
facemark = cv.face.createFacemarkLBF()
try:
    facemark.loadModel(cv.samples.findFile('lbfmodel.yaml'))
except cv.error:
    print("Model not found\nlbfmodel.yaml can be download at")
    print("https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml")
cascade = cv.CascadeClassifier(cv.samples.findFile('lbpcascade_frontalface_improved.xml'))
if cascade.empty() :
    print("cascade not found")
    exit()
faces = cascade.detectMultiScale(frame, 1.05,  3, cv.CASCADE_SCALE_IMAGE, (30, 30))
ok, landmarks = facemark.fit(frame, faces=faces)
cv.imshow("Image", frame)
for marks in landmarks:
    couleur = (random.randint(0,255),
               random.randint(0,255),
               random.randint(0,255))
    cv.face.drawFacemarks(frame, marks, couleur)
cv.imshow("Image Landmarks", frame)
cv.waitKey()
