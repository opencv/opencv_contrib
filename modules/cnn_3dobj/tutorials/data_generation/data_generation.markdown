Training data generation using Icosphere {#tutorial_data_generation}
=============

Goal
----

In this tutorial you will learn how to

-   Conduct a point cloud of camera view on sphere.
-   Generate training images using 3D model.

Code
----

@include cnn_3dobj/samples/sphereview_data.cpp

Explanation
-----------

Here is the general structure of the program:

-   Create a window.
    @code{.cpp}
    viz::Viz3d myWindow("Coordinate Frame");
    @endcode

-   Set window size as 64*64, we use this scale as default.
    @code{.cpp}
    myWindow.setWindowSize(Size(64,64));
    @endcode

-   Add coordinate axes.
    @code{.cpp}
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
    myWindow.setBackgroundColor(viz::Color::gray());
    myWindow.spin();
    @endcode

-   Create a Mesh widget, loading .ply models.
    @code{.cpp}
    viz::Mesh objmesh = viz::Mesh::load(plymodel);
    @endcode
-   Get the center of the generated mesh widget, cause some .ply files.
    @code{.cpp}
    Point3d cam_focal_point = ViewSphere.getCenter(objmesh.cloud);
    @endcode

-   Get the pose of the camera using makeCameraPoses.
    @code{.cpp}
    Affine3f cam_pose = viz::makeCameraPose(campos.at(pose)*radius+cam_focal_point, cam_focal_point, cam_y_dir*radius+cam_focal_point);
    @endcode

-   Get the transformation matrix from camera coordinate system to global.
    @code{.cpp}
    Affine3f transform = viz::makeTransformToGlobal(Vec3f(1.0f,0.0f,0.0f), Vec3f(0.0f,1.0f,0.0f), Vec3f(0.0f,0.0f,1.0f), campos.at(pose));
    viz::WMesh mesh_widget(objmesh);
    @endcode

-   Save screen shot as images.
    @code{.cpp}
    myWindow.saveScreenshot(filename);
    @endcode

-   Write images into binary files for further using in CNN training.
    @code{.cpp}
    ViewSphere.writeBinaryfile(filename, binaryPath, headerPath,(int)campos.size()*num_class, label_class, (int)(campos.at(pose).x*100), (int)(campos.at(pose).y*100), (int)(campos.at(pose).z*100), rgb_use);
    @endcode

Results
-------

Here is collection images created by this demo using 4 model.

![](images_all/1_8.png)
