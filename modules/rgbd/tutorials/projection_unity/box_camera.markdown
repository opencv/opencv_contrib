Projection mapping on boxes with camera {#tutorial_projection_unity_box_camera}
========

This tutorial explains how to map quad textures on planar surfaces using a webcam and a projector.

1. Stack some boxes.

![](img/box.png)

2. Open a calibration window by selecting *Window* -> *Calibration*.

![](img/menu.png)

3. Fill device ID and other parameters. Uncheck *Use Kinect*:

![](img/captureNoKinect.png)

4. Press *Start Scanning*.

5. Move *pattern* window to the projector screen, and press **f** key to make it fullscreen. If needed, press **w** to return to a window.

6. Place the camera on a projector. See *camera* window to make sure that the projected lines are distinguishable on the screen.

![](img/webcamMount.png)

7. Once the camera is placed, do not move it throughout the rest of the instructions.

8. Press space key to start structured lighting. Once started, wait until other windows shows up.  

![](img/structured.png)

9. See *correspondenceX* and *correspondenceY* windows. If the target object appeared as red/black, proceed to the Segmentation and UV mapping. If not, there are problems with lighting (room lighting and structured light intensity) or the target surface is too glossy.

10. Hit any key to close the app.

11. Import `projectorImage.png` to Unity.

![](img/imagePlane.png)

12. Place a HomographyPlane, make them transparent and drag the corners to align to the physical object.

![](img/transparentPlane.png)

13. Map texture.

![](img/texturedPlane.png)

14. Done!

![](img/projectedPlane.png)
