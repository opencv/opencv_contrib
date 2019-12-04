Scene Reconstruction {#tutorial_sfm_scene_reconstruction}
====================

Goal
----

In this tutorial you will learn how to use the reconstruction api for sparse reconstruction:

-   Load and file with a list of image paths.
-   Run libmv reconstruction pipeline.
-   Show obtained results using Viz.


Code
----

@include sfm/samples/scene_reconstruction.cpp

Explanation
-----------

Firstly, we need to load the file containing list of image paths in order to feed the reconstruction api:

@code{.cpp}
  /home/eriba/software/opencv_contrib/modules/sfm/samples/data/images/resized_IMG_2889.jpg
  /home/eriba/software/opencv_contrib/modules/sfm/samples/data/images/resized_IMG_2890.jpg
  /home/eriba/software/opencv_contrib/modules/sfm/samples/data/images/resized_IMG_2891.jpg
  /home/eriba/software/opencv_contrib/modules/sfm/samples/data/images/resized_IMG_2892.jpg

  ...

  int getdir(const string _filename, vector<string> &files)
  {
    ifstream myfile(_filename.c_str());
    if (!myfile.is_open()) {
      cout << "Unable to read file: " << _filename << endl;
      exit(0);
    } else {
      string line_str;
      while ( getline(myfile, line_str) )
        files.push_back(line_str);
    }
    return 1;
  }
@endcode

Secondly, the built container will be used to feed the reconstruction api. It is important outline that the estimated results must be stored in a vector<Mat>. In this
case is called the overloaded signature for real images which from the images, internally extracts and compute the sparse 2d features using DAISY descriptors in order to be matched using FlannBasedMatcher and build the tracks structure.

@code{.cpp}
  bool is_projective = true;
  vector<Mat> Rs_est, ts_est, points3d_estimated;
  reconstruct(images_paths, Rs_est, ts_est, K, points3d_estimated, is_projective);

  // Print output

  cout << "\n----------------------------\n" << endl;
  cout << "Reconstruction: " << endl;
  cout << "============================" << endl;
  cout << "Estimated 3D points: " << points3d_estimated.size() << endl;
  cout << "Estimated cameras: " << Rs_est.size() << endl;
  cout << "Refined intrinsics: " << endl << K << endl << endl;
@endcode

Finally, the obtained results will be shown in Viz.

Usage and Results
-----------------

In order to run this sample we need to specify the path to the image paths files, the focal length of the camera in addition to the center projection coordinates (in pixels).

**1. Middlebury temple**

Using following image sequence [1] and the followings camera parameters we can compute the sparse 3d reconstruction:

@code{.bash}
  ./example_sfm_scene_reconstruction image_paths_file.txt 800 400 225
@endcode

![](pics/temple_input.jpg)

The following picture shows the obtained camera motion in addition to the estimated sparse 3d reconstruction:

![](pics/temple_reconstruction.jpg)


**2. Sagrada Familia**

Using following image sequence [2] and the followings camera parameters we can compute the sparse 3d reconstruction:

@code{.bash}
  ./example_sfm_scene_reconstruction image_paths_file.txt 350 240 360
@endcode

![](pics/sagrada_familia_input.jpg)

The following picture shows the obtained camera motion in addition to the estimated sparse 3d reconstruction:

![](pics/sagrada_familia_reconstruction.jpg)

[1] [http://vision.middlebury.edu/mview/data](http://vision.middlebury.edu/mview/data)

[2] Penate Sanchez, A. and Moreno-Noguer, F. and Andrade Cetto, J. and Fleuret, F. (2014). LETHA: Learning from High Quality Inputs for 3D Pose Estimation in Low Quality Images. Proceedings of the International Conference on 3D vision (3DV).
[URL](http://www.iri.upc.edu/research/webprojects/pau/datasets/sagfam)
