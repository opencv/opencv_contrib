import numpy as np
import cv2 as cv

# aruco
adict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
cv.imshow("marker", adict.generateImageMarker(0, 400))

# random calibration data. your mileage may vary.
imsize = (800, 600)
K = cv.getDefaultNewCameraMatrix(np.diag([800, 800, 1]), imsize, True)

# AR scene
cv.ovis.addResourceLocation("packs/Sinbad.zip") # shipped with Ogre

win = cv.ovis.createWindow("arucoAR", imsize, flags=0)
win.setCameraIntrinsics(K, imsize)
win.createEntity("figure", "Sinbad.mesh", (0, 0, 5), (1.57, 0, 0))
win.createLightEntity("sun", (0, 0, 100))

# video capture
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, imsize[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, imsize[1])

while cv.ovis.waitKey(1) != 27:
    img = cap.read()[1]
    win.setBackground(img)
    corners, ids = cv.aruco.detectMarkers(img, adict)[:2]

    cv.waitKey(1)

    if ids is None:
        continue

    rvecs, tvecs = cv.aruco.estimatePoseSingleMarkers(corners, 5, K, None)[:2]
    win.setCameraPose(tvecs[0].ravel(), rvecs[0].ravel(), invert=True)
