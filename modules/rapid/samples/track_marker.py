import numpy as np
import cv2 as cv

# aruco config
adict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
cv.imshow("marker", cv.aruco.drawMarker(adict, 0, 400))
marker_len = 5

# rapid config
obj_points = np.float32([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]]) * marker_len
tris = np.int32([[0, 2, 1], [0, 3, 2]])  # note CCW order for culling
line_len = 10

# random calibration data. your mileage may vary.
imsize = (800, 600)
K = cv.getDefaultNewCameraMatrix(np.diag([800, 800, 1]), imsize, True)

# video capture
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, imsize[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, imsize[1])

rot, trans = None, None
while cv.waitKey(1) != 27:
    img = cap.read()[1]

    # detection with aruco
    if rot is None:
        corners, ids = cv.aruco.detectMarkers(img, adict)[:2]

        if ids is not None:
            rvecs, tvecs = cv.aruco.estimatePoseSingleMarkers(corners, marker_len, K, None)[:2]
            rot, trans = rvecs[0].ravel(), tvecs[0].ravel()

    # tracking and refinement with rapid
    if rot is not None:
        for i in range(5):  # multiple iterations
            ratio, rot, trans = cv.rapid.rapid(img, 40, line_len, obj_points, tris, K, rot, trans)[:3]
            if ratio < 0.8:
                # bad quality, force re-detect
                rot, trans = None, None
                break

    # drawing
    cv.putText(img, "detecting" if rot is None else "tracking", (0, 20), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255))
    if rot is not None:
        cv.drawFrameAxes(img, K, None, rot, trans, marker_len)
    cv.imshow("tracking", img)
