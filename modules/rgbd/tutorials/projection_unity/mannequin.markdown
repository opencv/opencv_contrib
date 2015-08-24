3D projection mapping on mannequin {#tutorial_projection_unity_mannequin}
========

Structured light
--------

1. Open a calibration window by selecting *Window* -> *Calibration*.
    ![](img/menu.png)

2. Fill device ID and other parameters. Check *Use Kinect*:
    ![](img/captureKinect.png)

3. Press *Start Scanning*.

4. Wait until windows show up.

5. Move *pattern* window to the projector screen, and press **f** key to make it fullscreen. If needed, press **w** to return to a window.

6. Place the depth camera in front of the projection target. See *camera* window to make sure that the projected lines are distinguishable on the screen. Also make sure that the target appears to be gray (not black) on the *depth* window.

7. Once the depth camera is placed, do not move it throughout the rest of the instructions.

8. Press space key to start structured lighting. Once started, wait until other windows shows up.

9. See *correspondenceX* and *correspondenceY* windows. If the target object appeared as red/black, proceed to the Segmentation and UV mapping. If not, there are problems with lighting (room lighting and structured light intensity) or the target surface is too glossy.

10. Hit any key to close the app.

Mesh generation
--------

1. Press *Start Meshing*:
    ![](img/captureKinect.png)

2. Wait until windows show up.

3. *clusterN* windows represent planes. *mesh_N_M* windows represent other objects (including the target object). 3D meshes are saved as *mesh_N_M.obj*.

4. Hit any key to close the app.

Unity3D projection mapping
--------

1. Open the Unity3D project and load *mapping.scene*.

2. Copy the 3D mesh file to the *Assets* folder.

3. Drag the 3D mesh file to the Hierarchy tab. Then, expand the *mesh_N_M* tree to reveal *default* GameObject.

4. Run the project.
    ![](img/projected.png)
