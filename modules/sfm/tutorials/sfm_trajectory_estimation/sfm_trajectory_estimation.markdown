Camera Motion Estimation {#tutorial_sfm_trajectory_estimation}
========================

Goal
----

In this tutorial you will learn how to use the reconstruction api for camera motion estimation:

-   Load a file with the tracked 2d points and build the container over all the frames.
-   Run libmv reconstruction pipeline.
-   Show obtained results using Viz.


Code
----

@include sfm/samples/trajectory_reconstruccion.cpp

Explanation
-----------

Firstly, we need to load the file containing the 2d points tracked over all the frames and construct the container to feed the reconstruction api. In this case the tracked 2d points will have the following structure, a vector of 2d points array, where each inner array represents a different frame. Every frame is composed by a list of 2d points which e.g. the first point in frame 1 is the same point in frame 2. If there is no point in a frame the assigned value will be (-1,-1):

@code{.cpp}
  /* Build the following structure data
   *
   *            frame1           frame2           frameN
   *  track1 | (x11,y11) | -> | (x12,y12) | -> | (x1N,y1N) |
   *  track2 | (x21,y11) | -> | (x22,y22) | -> | (x2N,y2N) |
   *  trackN | (xN1,yN1) | -> | (xN2,yN2) | -> | (xNN,yNN) |
   *
   *
   *  In case a marker (x,y) does not appear in a frame its
   *  values will be (-1,-1).
   */

   ...

  for (int i = 0; i < n_frames; ++i)
  {
    Mat_<double> frame(2, n_tracks);

    for (int j = 0; j < n_tracks; ++j)
    {
      frame(0,j) = tracks[j][i][0];
      frame(1,j) = tracks[j][i][1];
    }
    points2d.push_back(Mat(frame));
  }
@endcode

Secondly, the built container will be used to feed the reconstruction api. It is important outline that the estimated results must be stored in a vector<Mat>:

@code{.cpp}
  bool is_projective = true;
  vector<Mat> Rs_est, ts_est, points3d_estimated;
  reconstruct(points2d, Rs_est, ts_est, K, points3d_estimated, is_projective);

  // Print output

  cout << "\n----------------------------\n" << endl;
  cout << "Reconstruction: " << endl;
  cout << "============================" << endl;
  cout << "Estimated 3D points: " << points3d_estimated.size() << endl;
  cout << "Estimated cameras: " << Rs_est.size() << endl;
  cout << "Refined intrinsics: " << endl << K << endl << endl;
@endcode

Finally, the obtained results will be shown in Viz, in this case reproducing the camera with an oscillation effect.

Usage and Results
-----------------

In order to run this sample we need to specify the path to the tracked points file, the focal lenght of the camera in addition to the center projection coordinates (in pixels). You can find a sample file in samples/data/desktop_trakcks.txt

@code{.bash}
  ./example_sfm_trajectory_reconstruction desktop_tracks.txt 1914 640 360
@endcode

The following picture shows the obtained camera motion obtained from the tracked 2d points:

![](pics/desktop_trajectory.png)