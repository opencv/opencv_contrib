Projection mapping on boxes without camera {#tutorial_projection_unity_box_nocamera}
========


This tutorial explains how to manually map quad textures on planar surfaces.

1. Stack some boxes.
	![](img/box.png)

2. Move an existing HomographyPlane (or add drag *Prefabs/HomographyPlane.prefab* to the *Planes* in *Hierarchy*).

3. Align the plane corners by looking at the physical object.
	![](img/manualAlignment.png)

4. Map a texture by dragging a material (e.g., *Materials/defaultMat.mat*) to the object in the *Scene*. You may change the Shader (use either *Custom/FalseDepth* or *Custom/FalseDepthUnlit*) or its parameters to configure the appearance.
	![](img/texturedPlane.png)

5. Done!
	![](img/projectedPlane.png)
