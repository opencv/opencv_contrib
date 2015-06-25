Depth map clustering and projection mapping {#tutorial_rgbd_clustering}
=============================================================

Projection mapping (spatical augmented reality) is a technique to augment a scene using a projector ([ref](http://www.creativebloq.com/video/projection-mapping-912849)).
Often in large scale projection mapping, a target object is modeled by hand in 3D and the model is manually aligned to the target object.
However, since commodity depth sensors are available, such a model can be generated automatically.
Moreover, projector-model alignment can be facilitated by projector-camera calibration.
Thus, the motivation of this tutorial is to achive smart projection mapping using computer vision approaches.
Although most of the projection mapping tutorials introduce projection on planar surfaces by quad warping
(homography), it does not give much flexibility nor exploit the depth cue. Instead, we demonstrate a method to calibrate projector-camera
and to automatically segment background objects for projection target extraction. Finally, using Unity3D, a texture can be mapped on each 3D object
to be projected on physical objects. To simplify the problem, we assume that the scene consists only of projection targets and flat objects
(e.g., tabletop, wall).

Goal
----

In this tutorial you will learn how to:

-   calibrate a projector and a depth camera;
-   segment different 3D objects in a depth map;
-   map texture in Unity3D.

Projector-camera calibration and depth map acquisition
-------------

To be implemented/documented.

Depth map clustering code explanation
-------------

Please refer to the original tutorial source code in file
*opencv_folder/samples/cluster_projection.cpp*.

@note do not forget that the clustering API is included in the following namespace: cv::rgbd

Here is a code explanation :

Depth map clustering is present in the rgbd package and a simple include allows to use it. You
can rather use the specific header : *opencv2/rgbd.hpp* if you prefer but then include the
other required opencv modules : *opencv2/core.hpp* and *opencv2/highgui.hpp*

RgbdCluster includes a mask image (`Mat mask`), an original depth map (`Mat depth, points3d`),
and a vector of valid points (`std::vector<RgbdPoint> points`). This enables developers to use a cluster
as both an image to apply OpenCV functions and a point cloud to process as a 3D mesh.

The code starts with importing data,

@code{.cpp}
    // read depth data from libfreenect
    cv::FileStorage file("rgbd.txt", cv::FileStorage::READ);
    
    file["depth"] >> depth;
    file["zeroPlanePixelSize"] >> pixelSize;
    file["zeroPlaneDistance"] >> refDistance;
    depth = depth * 0.001f; // libfreenect is in [mm]
@endcode

Import depth and sensor data from libfreenect. Since png format is 8-bit and a depth map can be
16-bit or more, we prefer storing raw depth data. zeroPlanePixelSize and zeroPlaneDistance
defines the depth camera focal length (libfreenect specific).

@code{.cpp}
    RgbdCluster mainCluster;
    mainCluster.points3d = points3d;
    mainCluster.depth = frame->depth;
    vector<RgbdCluster> clusters;
    planarSegmentation(mainCluster, clusters);
    deleteEmptyClusters(clusters);
@endcode

planarSegmentation(...) wraps RgbdPlane and generates a vector of masked RGBD images, which are stored in clusters.
The first cluster includes a binary mask of the most dominant plane, the second cluster is the second dominant plane, and so on.
If the cluster's bPlane is false, the cluster includes a mask of the remaining part.
Since RgbdPlane extracts a plane at z = 0 (pixels with unknown depth are set to z = 0),
this plane must be deleted by deleteEmptyClusters(...).

@code{.cpp}
        vector<RgbdCluster> smallClusters;
        euclideanClustering(clusters.at(i), smallClusters);
@endcode

The last element of the clusters, which is non-planar objects, can be further split by euclideanClustering(...).
This function wraps connectedcomponents(...) to simply separate the blobs.

@code{.cpp}
            smallClusters.at(j).unwrapTexCoord();
            smallClusters.at(j).save(to_string(i) + to_string(j) + "mesh.obj");
@endcode

Each point in the cluster is assigned a (u, v) value, which is a texture coordinate. The LSCM
algorithm is used so that the 3D distances between neighboring points will be roughly preserved
in the UV coordinate (not implemented yet, this is done through Blender).
Finally, the mesh with the texture coordinate is saved to be imported to Unity3D.

Projection mapping using Unity3D
-------------

Please refer to the original tutorial project folder in
*opencv_folder/samples/unity_mapping*.

Follow the instructions to import data:

1. Open *mapping.scene*, which already includes a customized camera.

2. Select "Main Camera," and load calibration data by clicking "Load Calibration" in the inspector.
Camera parameters will be automatically applied to the "Main Camera."

3. Copy 3D mesh files generated in the previous section to the *Assets/Models* folder. Then, drag the mesh file into the hierarchy window. Open the mesh tree.

4. Move to *Assets/Textures* folder and drag a texture to "default" in the mesh hierarchy.

5. Click on the play button to preview. At this point, the preview window cannot be fullscreen and the aspect ratio is
perhaps different from that of the projector. Therefore, the rendered image will not be aligned to the physical scene.
Unfortunately, Unity3D does not support fullscreen preview, so an executable file must be generated to run the program in fullscreen.
Select File->Build and Run.
