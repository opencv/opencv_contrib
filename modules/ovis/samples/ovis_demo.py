import cv2
import numpy as np

# add some external resources
cv2.ovis.addResourceLocation("packs/Sinbad.zip")

# camera intrinsics
imsize = (800, 600)
K = np.diag([800, 800, 1])
K[:2, 2] = (400, 100) # offset pp

# observer scene
owin = cv2.ovis.createWindow("VR", imsize)
cv2.ovis.createGridMesh("ground", (10, 10), (10, 10))
owin.createEntity("ground", "ground", rot=(1.57, 0, 0))
owin.createCameraEntity("cam", K, imsize, 5)
owin.createEntity("figure", "Sinbad.mesh", (0, -5, 0))  # externally defined mesh
owin.createLightEntity("sun", (0, 0, -100))

# interaction scene
iwin = cv2.ovis.createWindow("AR", imsize, cv2.ovis.SCENE_SEPERATE | cv2.ovis.SCENE_INTERACTIVE)
iwin.createEntity("figure", "Sinbad.mesh", (0, -5, 0))
iwin.createLightEntity("sun", (0, 0, -100))
iwin.setCameraIntrinsics(K, imsize)

while cv2.ovis.renderOneFrame():
    R, t = iwin.getCameraPose()
    owin.setEntityPose("cam", t, R)
