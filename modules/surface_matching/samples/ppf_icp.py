import cv2
import numpy as np

def rotation(theta):
    tx, ty, tz = theta

    Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, -np.sin(ty)], [0, 1, 0], [np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]])

    return np.dot(Rx, np.dot(Ry, Rz))

width = 20
height = 10
max_deg = np.pi / 12

cloud, rotated_cloud = [None]*3, [None]*3
retval, residual, pose = [None]*3, [None]*3, [None]*3
noise = np.random.normal(0.0, 0.1, height * width * 3).reshape((-1, 3))
noise2 = np.random.normal(0.0, 1.0, height * width)

x, y = np.meshgrid(
    range(-width//2, width//2),
    range(-height//2, height//2),
    sparse=False, indexing='xy'
)
z = np.zeros((height, width))

cloud[0] = np.dstack((x, y, z)).reshape((-1, 3)).astype(np.float32)
cloud[1] = noise.astype(np.float32) + cloud[0]
cloud[2] = cloud[1]
cloud[2][:, 2] += noise2.astype(np.float32)

R = rotation([
    0, #np.random.uniform(-max_deg, max_deg),
    np.random.uniform(-max_deg, max_deg),
    0, #np.random.uniform(-max_deg, max_deg)
])
t = np.zeros((3, 1))
Rt = np.vstack((
    np.hstack((R, t)),
    np.array([0, 0, 0, 1])
)).astype(np.float32)

icp = cv2.ppf_match_3d_ICP(100)

I = np.eye(4)
print("Unaligned error:\t%.6f" % np.linalg.norm(I - Rt))
for i in range(3):
    rotated_cloud[i] = np.matmul(Rt[0:3,0:3], cloud[i].T).T + Rt[:3,3].T
    retval[i], residual[i], pose[i] = icp.registerModelToScene(rotated_cloud[i], cloud[i])
    print("ICP error:\t\t%.6f" % np.linalg.norm(I - np.matmul(pose[0], Rt)))
