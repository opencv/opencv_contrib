import numpy as np
import cv2 as cv

v = cv.viz.Viz3d_create("Viz Demo")

print("First event loop is over")
v.spin()
print("Second event loop is over")
v.spinOnce(1, True)
while not v.wasStopped():
    v.spinOnce(1, True)
print("Last event loop is over")
