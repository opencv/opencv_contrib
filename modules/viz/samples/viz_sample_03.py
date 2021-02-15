#exec(open(r'G:\Lib\opencv_contrib\modules\viz\samples\viz_sample_03.py').read())
import os
os.add_dll_directory(r'G:\Lib\install\opencv\x64\vc15\bin')
os.add_dll_directory(r'G:\Lib\install\vtk\bin')
os.add_dll_directory(r'G:\Lib\install\ceres-solver\bin')
os.add_dll_directory(r'G:\Lib\install\glog\bin')
os.add_dll_directory(r'F:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')
os.add_dll_directory(r'F:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2')

import cv2 as cv
import numpy as np

def affine_translate(v):
    pose = np.zeros(shape=(4,4),dtype=np.float64)
    pose[0, 3] = 1
    pose[1, 3] = 1
    pose[1, 3] = 1
    pose[0:3, 0] = v[0]
    pose[0:3, 1] = v[1]
    pose[0:3, 2] = v[2]
    return pose

def load_bunny():
    with open(r'G:\Lib\opencv_contrib\modules\viz\samples\data\bunny.ply', 'r') as f:
        s = f.read()
    ligne = s.split('\n')
    if len(ligne) == 5753:
        pts3d = np.zeros(shape=(1,1889,3), dtype=np.float32)
        pts3d_c = 255 * np.ones(shape=(1,1889,3), dtype=np.uint8)
        pts3d_n = np.ones(shape=(1,1889,3), dtype=np.float32)
        for idx in range(12,1889):
            d = ligne[idx].split(' ')
            pts3d[0,idx-12,:] = (float(d[0]), float(d[1]), float(d[2]))
    pts3d = 5 * pts3d
    return cv.viz_PyWCloud(pts3d)

myWindow = cv.viz_Viz3d("Coordinate Frame")
axe = cv.viz_PyWCoordinateSystem()
myWindow.showWidget("axe",axe)

cam_pos =  (3.0, 3.0, 3.0)
cam_focal_point = (3.0,3.0,2.0)
cam_y_dir = (-1.0,0.0,0.0)
cam_pose = cv.viz.makeCameraPosePy(cam_pos, cam_focal_point, cam_y_dir)
print("OK")
transform = cv.viz.makeTransformToGlobalPy((0.0,-1.0,0.0), (-1.0,0.0,0.0), (0.0,0.0,-1.0), cam_pos)
pw_bunny = load_bunny()
cloud_pose = cv.viz_PyAffine3()
cloud_pose = cloud_pose.translate((0, 0, 3))
cloud_pose_global = transform.product(cloud_pose)

cpw = cv.viz_PyWCameraPosition(0.5)
cpw_frustum = cv.viz_PyWCameraPosition(0.3)
myWindow.showWidget("CPW", cpw);
myWindow.showWidget("CPW_FRUSTUM", cpw_frustum)
myWindow.setViewerPosePy(cam_pose)
myWindow.showWidget("bunny", pw_bunny, cloud_pose_global)
#myWindow.setWidgetPosePy("bunny")
myWindow.spin();
print("Last event loop is over")
