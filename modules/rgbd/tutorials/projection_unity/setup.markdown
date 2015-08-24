Setup Unity3D project {#tutorial_projection_unity_setup}
========

Before proceeding to the tutorial, follow these instructions to setup Unity3D for projection mapping.

1. Align the monitor and projector displays side-by-side. Display mirroring is currently not supported.
	![](img/screenOsSettings.png)

2. Open *opencv_contrib/modules/rgbd/samples/UnityMapping* in Unity3D.

3. Load *Scenes/mapping.unity* if not opened yet.

4. Navigate to *Hierarchy* -> *Projector Viewport* -> *Main Camera*. In the *Inspector*, fill *Projector width* and *Projector Height*.

5. Open the *Calibration* window by selecting *Window* -> *Calibration*.
	![](img/menu.png)

6. In the *Calibration* window, press *Select Folder Containing Executables* to select the folder with OpenCV executables, then *Select Assets Folder* to choose *Assets/*. Also you may need to change *Monitor Width*, *Projector Width* and *Projector Height* (projector resolution).
	![](img/calibrationManager.png)

7. Open the Screen Setting window by selecting *Window* -> *Screen Settings*.
	![](img/menu.png)

8. Fill *Monitor Width* (width of your monitor in pixels), *Projector Width* and *Projector Height* (projector resolution).
	![](img/screenSettings.png)

9. Keep this window floating all the time; this helps when the game window accidentally covered the monitor.

10. When you're done, check *Go Fullscreen*. You can check/uncheck it whenever you want, regardless of the game is playing or not.

11. You're ready to begin the tutorials!
