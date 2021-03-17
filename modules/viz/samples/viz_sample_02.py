import numpy as np
import cv2 as cv

my_window = cv.viz_Viz3d("Coordinate Frame")

axe = cv.viz_PyWCoordinateSystem()
axis = cv.viz_PyWLine((-1.0,-1.0,-1.0), (1.0,1.0,1.0), cv.viz_PyColor().green())
axis.setRenderingProperty(cv.viz.LINE_WIDTH, 4.0);
my_window.showWidget("axe",axis)
plan = cv.viz_PyWPlane((-1.0,-1.0,-1.0), (1.0,.0,.0), (-.0,.0,-1.0))
#my_window.showWidget("plan", plan)
cube = cv.viz_PyWCube((0.5,0.5,0.0), (0.0,0.0,-0.5), True, cv.viz_PyColor().blue())

#my_window.showWidget("Cube Widget",cube)
pi = np.arccos(-1)
print("First event loop is over")
my_window.spin()
print("Second event loop is over")
my_window.spinOnce(1, True)
translation_phase = 0.0
translation = 0.0
rot_mat = np.zeros(shape=(3, 3), dtype=np.float32)
rot_vec = np.zeros(shape=(1,3),dtype=np.float32)
while not my_window.wasStopped():
    rot_vec[0, 0] += np.pi * 0.01
    rot_vec[0, 1] += np.pi * 0.01
    rot_vec[0, 2] += np.pi * 0.01
    translation_phase += pi * 0.01
    translation = np.sin(translation_phase)
    pose = cv.viz_PyAffine3(rot_vec, (translation, translation, translation))
    my_window.setWidgetPosePy("Cube Widget", pose)
    my_window.spinOnce(1, True)
print("Last event loop is over")
