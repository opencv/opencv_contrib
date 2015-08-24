Projection mapping on boxes with camera {#tutorial_projection_unity_box_camera}
========

This tutorial explains how to map quad textures on planar surfaces using a webcam and a projector.

1. Stack some boxes.
	![](img/box.png)

2. Open a calibration window by selecting *Window* -> *Calibration*.
	![](img/menu.png)

3. Fill device ID and other parameters. Uncheck *Use Kinect*:
	![](img/captureNoKinect.png)

4. Connect a webcam and press *Start Scanning*.

5. A stripe pattern will show up in fullscreen on the projector display. If not, click on the stripe pattern window, press **w** to return from fullscreen. Then, drag the window to the projector screen, and press **f** to make it fullscreen again.

6. Place the camera near the projector. See *camera* window to make sure that the projected lines are distinguishable on the screen. The projector and camera should be facing the same direction.
	![](img/webcamMount.png)

7. Once the camera is placed, do not move it throughout the rest of the instructions.

8. Press space key to start structured lighting. Once started, wait until other windows shows up.
	![](img/structured.png)

9. See *correspondenceX* and *correspondenceY* windows. If the target object appeared as red/black, proceed to the next step. If not, there might be problems with lighting (room lighting and structured light intensity) or the target surface is too glossy. Try to make a shade on the object and/or change the *Light Intensity* in the *Calibraiton* window. Then, hit any key to close the windows and go back to the step 3.

10. Hit any key to close the windows. Scanned data will be automatically loaded to the Unity scene.

11. Move an existing HomographyPlane (or add drag *Prefabs/HomographyPlane.prefab* to the *Planes* in *Hierarchy*). Make them transparent by dragging *Materials/Transparent.mat* on the planes.

12. Move the corners to align to the corners in the background image.
	![](img/transparentPlane.png)

13. Map a texture by dragging a material (e.g., *Materials/defaultMat.mat*) to the object in the *Scene*. You may change the Shader (use either *Custom/FalseDepth* or *Custom/FalseDepthUnlit*) or its parameters to configure the appearance.
	![](img/texturedPlane.png)

14. Done!
	![](img/projectedPlane.png)
