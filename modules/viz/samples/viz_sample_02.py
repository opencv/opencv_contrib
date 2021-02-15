import os
os.add_dll_directory(r'G:\Lib\install\opencv\x64\vc15\bin')
os.add_dll_directory(r'G:\Lib\install\vtk\bin')
os.add_dll_directory(r'G:\Lib\install\ceres-solver\bin')
os.add_dll_directory(r'G:\Lib\install\glog\bin')
os.add_dll_directory(r'F:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')
os.add_dll_directory(r'F:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2')

import cv2 as cv
import numpy as np

my_window = cv.viz_Viz3d("Coordinate Frame")

pw = cv.viz_ParamWidget()
axe = cv.viz_PyWCoordinateSystem()
axis = cv.viz_PyWLine((-1.0,-1.0,-1.0), (1.0,1.0,1.0))
axis.setRenderingProperty(cv.viz_LINE_WIDTH, 4.0);
my_window.showWidget("axe",axis)
plan = cv.viz_PyWPlane((-1.0,-1.0,-1.0), (1.0,.0,.0), (-.0,.0,-1.0))
v.showWidget("plan", plan)

v.showWidget(p_cube)
pi = np.arccos(-1)
print("First event loop is over")
v.spin()
print("Second event loop is over")
v.spinOnce(1, True)
translation_phase = 0.0
translation = 0.0
rot_mat = np.zeros(shape=(3, 3), dtype=np.float32)
p_cube.pose = np.zeros(shape=(4, 4), dtype=np.float32)
p_cube.pose[3, 3] = 1
while not v.wasStopped():
    p_cube.rot_vec[0, 0] += pi * 0.01
    p_cube.rot_vec[0, 1] += pi * 0.01
    p_cube.rot_vec[0, 2] += pi * 0.01
    translation_phase += pi * 0.01
    translation = np.sin(translation_phase)
    cv.Rodrigues(p_cube.rot_vec, rot_mat)
    p_cube.pose[0:3,0:3] = rot_mat
    p_cube.pose[0:3,3] = translation
    v.setWidgetPose(p_cube)
    v.spinOnce(1, True)
print("Last event loop is over")
