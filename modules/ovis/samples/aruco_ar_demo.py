import cv2
import numpy as np

# aruco
adict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
cv2.imshow("marker", cv2.aruco.drawMarker(adict, 0, 400))

# random calibration data. your mileage may vary.
imsize = (800, 600)
K = cv2.getDefaultNewCameraMatrix(np.diag([800, 800, 1]), imsize, True)

# AR scene
cv2.ovis.addResourceLocation("packs/Sinbad.zip") # shipped with Ogre

win = cv2.ovis.createWindow("arucoAR", imsize, flags=0)
win.createEntity("figure", "Sinbad.mesh", (0, 0, -5), (-1.57, 0, 0))
win.createLightEntity("sun", (0, 0, -100))

# video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, imsize[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imsize[1])

while cv2.ovis.renderOneFrame():
    img = cap.read()[1]
    win.setBackground(img)
    corners, ids = cv2.aruco.detectMarkers(img, adict)[:2]

    cv2.waitKey(1)

    if ids is None:
        continue

    rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners, 5, K, None)[:2]
    win.setCameraPose(tvecs[0].ravel(), rvecs[0].ravel(), invert=True)
