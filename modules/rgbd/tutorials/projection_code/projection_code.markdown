Depth map clustering and projection mapping {#tutorial_projection_code}
=============================================================

Projection mapping (spatical augmented reality) is a technique to augment a scene using a projector ([ref](http://www.creativebloq.com/video/projection-mapping-912849)).
Often in large scale projection mapping, a target object is modeled by hand in 3D and the model is manually aligned to the target object.
However, since commodity depth sensors are available, such a model can be generated automatically.
Moreover, projector-model alignment can be facilitated by projector-camera calibration.
Thus, the motivation of this tutorial is to achieve smart projection mapping using computer vision approaches.
Although most of the projection mapping tutorials introduce projection on planar surfaces by quad warping
(homography), it does not give much flexibility nor exploit the depth cue. Instead, we demonstrate a method to calibrate projector-camera and to automatically segment background objects for projection target extraction. Finally, using Unity3D, a texture can be mapped on each 3D object to be projected on physical objects. To simplify the problem, we assume that the scene consists only of projection targets and flat objects (e.g., tabletop, wall).

Goal
----

In this tutorial you will learn how to:

-   calibrate a projector and a depth camera;
-   segment different 3D objects in a depth map and assign UV texture coordinate;
-   map texture in Unity3D for projection mapping.

Sensor-projector calibration
-------------

Please refer to the original tutorial source code in file
*opencv_folder/samples/sensor_projector_calibration.cpp*.

@note In addition to rgbd API, we use structured lighting API, which is included in the following namespace: cv::structured_light

In this section, a calibration how-to is explained in parallel with a code explanation. To begin with, you need to prepare a OpenNI2-compatible sensor (Kinect V1 and Primesense sensor, etc.) and a projector, preferably attached to each other. The sensor-projector unit has to face a scene that consists of more than one surfaces; for example, a corner of a room with some boxes can work, but the calibration will fail if the scene is only a flat wall.

Let's launch the calibrator.

@code{.cpp}

    VideoCapture capture(CAP_OPENNI2);

@endcode

Open an OpenNI2 device.

@code{.cpp}

    // initialize gray coding
    GrayCodePattern::Params params;
    params.width = 1024;
    params.height = 768;
    Ptr<GrayCodePattern> pattern = GrayCodePattern::create(params);

    vector<Mat> patternImages;
    pattern->generate(patternImages, Scalar(0, 0, 0), Scalar(100, 100, 100));

@endcode

Generate gray code patterns.

@code{.cpp}

    // window placement; wait for user
    while (true)
    {
        int key = waitKey(30);
        if (key == 'f')
        {
            // TODO: 1px border when fullscreen on Windows (Surface?)
            setWindowProperty(window, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
        }
        else if (key == 'w')
        {
            setWindowProperty(window, WND_PROP_FULLSCREEN, WINDOW_NORMAL);
        }
        else if (key == ' ')
        {
            break;
        }
    }

@endcode

Wait for a user input. The user can move the window to the projector screen and switch between fullscreen and window mode by pressing 'f' and 'w' keys. When the space key is pressed, move on to structured light projection.

@code{.cpp}

    for (size_t i = 0; i < patternImages.size(); i++) {
        imshow(window, patternImages.at(i));

        waitKey(500);

        capture.grab();

        ...
    }

@endcode

Initiate projection of gray-code structured light. After projecting every code, the capture will take a corresponding color image and converts to a gray image for decoding. Also a depth image is taken, which will be averaged for smoothing.

@code{.cpp}

    // decode
    vector<Point3f> objectPoints;
    vector<Point2f> imagePoints;
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            if (frame->mask.at<uchar>(y, x) == 0)
            {
                continue;
            }

            Point point;
            if (pattern->getProjPixel(cameraImages, x, y, point) != true)
            {
                Point3f p3d = frame->points3d.at<Point3f>(y, x);
                objectPoints.push_back(p3d);
                imagePoints.push_back(Point2f(point.x, point.y));
            }
        }
    }

@endcode

The code of each pixel in the sensor camera image is decoded by `getProjPixel(...)` to find the corresponding projector pixel. If the depth value (3D coordinate) of the pixel is available and the code can be decoded, a pair of the 3D coordinate and projector pixel will be stored to `objectPoints` and `imagePoints`.

@code{.cpp}

    solvePnPRansac(objectPoints, imagePoints, projectorMatrix, distCoeffs, rvec, tvec, true, 100, 3, 0.99);

@endcode

Using the 3D-2D point pairs, `solvePnPRansac(...)` will find the optimal pose (position and orientation) of the projector relative to the sensor (specifically, its color camera). Currently, the focal length and principal point of the projector are assumed to be known beforehand.

@code{.cpp}

    // file output
    cout << extrinsics << endl;

    FileStorage fs("calibration.xml", FileStorage::WRITE);
    fs << "ProjectorCameraEnsemble";
    fs << "{";
    {
        ...
    }
    fs << "}";

@endcode

The calibration results are saved to an XML file. It looks daunting, but the XML hierarchy makes it easy to load by C# XML serializer, which will be called in Unity3D. The calibration is done, so let's capture the scene and find interesting objects in the scene for projection mapping.

Depth map clustering
-------------

### Depth map clustering API

@note do not forget that the clustering API is included in the following namespace: cv::rgbd

Depth map clustering is present in the rgbd package and a simple include allows to use it. You can rather use the specific header : *opencv2/rgbd.hpp* if you prefer but then include the other required opencv modules : *opencv2/core.hpp* and *opencv2/highgui.hpp*

RgbdCluster includes a mask image (`Mat mask`), an original depth map (`Mat depth, points3d`), and a vector of valid points (`std::vector<RgbdPoint> points`). This enables developers to use a cluster as both an image to apply OpenCV functions and a point cloud to process as a 3D mesh.


Please refer to the original tutorial source code in file
*opencv_folder/samples/cluster_projection.cpp*.

### Tutorial and code explanation

We assume that you have completed sensor-projector calibration. Now, you may move the sensor-projector unit to the scene you want to project. The scene should consists only of projection targets and flat objects (e.g., tabletop, wall).

The code starts with capturing a depth image,

@code{.cpp}

    while(true)
    {
        capture.grab();

        capture.retrieve(image, CAP_OPENNI_BGR_IMAGE);
        imshow("Color", image);

        capture.retrieve(depth, CAP_OPENNI_DEPTH_MAP);
        imshow("Depth", depth * 8);

        if (waitKey(30) >= 0)
            break;
    }

@endcode

The live feed of the color camera and depth sensor are shown in separate windows. The user can interrupt any time by pressing a key to start segmentation.

@code{.cpp}

    RgbdClusterMesh mainCluster(frame);
    vector<RgbdClusterMesh> clusters;
    planarSegmentation(mainCluster, clusters);
    deleteEmptyClusters(clusters);

@endcode

`planarSegmentation(...) wraps `RgbdPlane` and generates a vector of masked RGBD images, which are stored in clusters.
The first cluster includes a binary mask of the most dominant plane, the second cluster is the second dominant plane, and so on.
If the cluster's `bPlane` is false, the cluster includes a mask of the remaining part.
Since `RgbdPlane` extracts a dummy plane at z = 0 (pixels with unknown depth are set to z = 0) as well, this plane must be deleted by deleteEmptyClusters(...).

@code{.cpp}

        vector<RgbdCluster> smallClusters;
        euclideanClustering(clusters.at(i), smallClusters);

@endcode

The last element of the clusters, which is non-planar objects, can be further split by `euclideanClustering(...)`.
This function wraps `connectedcomponents(...)` to simply separate the blobs.

@code{.cpp}

            // downsample by 0.5x
            smallClusters.at(j).increment_step = 2;
            smallClusters.at(j).calculatePoints();
            smallClusters.at(j).unwrapTexCoord();
            smallClusters.at(j).save(ss.str() + ".obj");
            smallClusters.at(j).save(ss.str() + ".ply");

@endcode

Each point in the cluster is assigned a (u, v) value, which is a texture coordinate.
The LSCM algorithm is used so that the 3D distances between neighboring points will be roughly preserved in the UV coordinate (not implemented yet, this is done through Blender).
Finally, the mesh with the texture coordinate is saved to be imported to Unity3D.

Projection mapping using Unity3D
-------------

Please refer to the original tutorial project folder in
*opencv_folder/samples/unity_mapping*.

Follow the instructions to import the calibration data and targets:

1. Open *mapping.scene*, which already includes a customized camera.

2. Select "Main Camera," and load the calibration data by clicking "Load Calibration" in the inspector.
Camera parameters will be automatically applied to the "Main Camera."

3. Copy 3D mesh files generated in the previous section to the *Assets/Models* folder. Then, drag the mesh file into the hierarchy window. Open the mesh tree.

4. Move to *Assets/Textures* folder and drag a texture to "default" in the mesh hierarchy.

5. Click on the play button to preview. At this point, the preview window cannot be fullscreen and the aspect ratio is
perhaps different from that of the projector. Therefore, the rendered image will not be aligned to the physical scene.
Unfortunately, Unity3D does not support fullscreen preview, so an executable file must be generated to run the program in fullscreen.
Select File->Build and Run. When the compilation is done and a dialog popped up, select fullscreen and the projector screen.
