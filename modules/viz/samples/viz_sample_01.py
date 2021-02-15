import os
os.add_dll_directory(r'G:\Lib\install\opencv\x64\vc15\bin')
os.add_dll_directory(r'G:\Lib\install\vtk\bin')
os.add_dll_directory(r'G:\Lib\install\ceres-solver\bin')
os.add_dll_directory(r'G:\Lib\install\glog\bin')
os.add_dll_directory(r'F:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')
os.add_dll_directory(r'F:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2')

import cv2 as cv
import numpy as np

v = cv.viz.Viz3d_create("Viz Demo")

print("First event loop is over")
v.spin()
print("Second event loop is over")
v.spinOnce(1, True)
while not v.wasStopped():
    v.spinOnce(1, True)
print("Last event loop is over")
