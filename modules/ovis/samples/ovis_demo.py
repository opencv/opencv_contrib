import numpy as np
import cv2 as cv

# add some external resources
cv.ovis.addResourceLocation("packs/Sinbad.zip")

# camera intrinsics
imsize = (800, 600)
K = np.diag([800, 800, 1])
K[:2, 2] = (400, 500) # offset pp

# observer scene
owin = cv.ovis.createWindow("VR", imsize)
cv.ovis.createGridMesh("ground", (10, 10), (10, 10))
owin.createEntity("ground", "ground", rot=(1.57, 0, 0))
owin.createCameraEntity("cam", K, imsize, 5)
owin.createEntity("sinbad", "Sinbad.mesh", tvec=(0, -5, 0), rot=(np.pi, 0, 0))  # externally defined mesh
owin.createLightEntity("sun", (0, 0, -100))

# setup and play idle animation
owin.setEntityProperty("sinbad", cv.ovis.ENTITY_ANIMBLEND_MODE, 1) # 1 = cumulative
owin.playEntityAnimation("sinbad", "IdleBase")
owin.playEntityAnimation("sinbad", "IdleTop")

# interaction scene
iwin = cv.ovis.createWindow("AR", imsize, cv.ovis.SCENE_SEPARATE | cv.ovis.SCENE_INTERACTIVE)
iwin.createEntity("sinbad", "Sinbad.mesh", tvec=(0, -5, 0), rot=(np.pi, 0, 0))
iwin.createLightEntity("sun", (0, 0, -100))
iwin.setCameraIntrinsics(K, imsize)

while cv.ovis.waitKey(1) != 27:
    R, t = iwin.getCameraPose()
    owin.setEntityPose("cam", t, R)

del iwin # must be destroyed in reverse creation order