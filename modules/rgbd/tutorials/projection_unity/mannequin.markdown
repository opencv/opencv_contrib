3D projection mapping on mannequin {#tutorial_projection_unity_mannequin}
========

Structured light
--------

1. Open a calibration window by selecting *Window* -> *Calibration*.
    ![](img/menu.png)

2. Fill device ID and other parameters. Check *Use Kinect*:
    ![](img/captureKinect.png)

3. Connect a Kinect and press *Start Scanning*.

4. Wait until windows show up.

5. A stripe pattern will show up in fullscreen on the projector display. If not, click on the stripe pattern window, press **w** to return from fullscreen. Then, drag the window to the projector screen, and press **f** to make it fullscreen again.

6. Place the depth camera in front of the projection target. See *camera* window to make sure that the projected lines are distinguishable on the screen. Also make sure that the target appears to be gray (not black) on the *depth* window.

7. Once the depth camera is placed, do not move it throughout the rest of the instructions.

8. Press space key to start structured lighting. Once started, wait until other windows shows up.

9. See *correspondenceX* and *correspondenceY* windows. If the target object appeared as red/black, proceed to the next step. If not, there might be problems with lighting (room lighting and structured light intensity) or the target surface is too glossy. Try to make a shade on the object and/or change the *Light Intensity* in the *Calibraiton* window. Then, hit any key to close the windows and go back to the step 3.

10. Hit any key to close the app.

Mesh generation
--------

1. Press *Start Meshing*:
    ![](img/captureKinect.png)

2. Wait until windows show up and disappear.

3. Move *Assets/Opencv/mesh_N_M.obj* to the *Hierarchy*. Make sure its *Position* and *Rotation* are 0 and *Scale* is 1.

4. Run the project.
    ![](img/projected.png)
