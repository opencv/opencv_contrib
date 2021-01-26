#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::sfm;

static void help() {
  cout
      << "\n------------------------------------------------------------------------------------\n"
      << " This program shows the multiview reconstruction capabilities in the \n"
      << " OpenCV Structure From Motion (SFM) module.\n"
      << " It reconstruct a scene from a set of 2D images \n"
      << " and add color to each voxel \n"
      << " Usage:\n"
      << "        example_sfm_scene_reconstruction_in_color <path_to_file> <f> <cx> <cy>\n"
      << " where: path_to_file is the file absolute path into your system which contains\n"
      << "        the list of images to use for reconstruction. \n"
      << "        f  is the focal length in pixels. \n"
      << "        cx is the image principal point x coordinates in pixels. \n"
      << "        cy is the image principal point y coordinates in pixels. \n"
      << "------------------------------------------------------------------------------------\n\n"
      << endl;
}


static int getdir(const string _filename, vector<String> &files)
{
  ifstream myfile(_filename.c_str());
  if (!myfile.is_open()) {
    cout << "Unable to read file: " << _filename << endl;
    exit(0);
  } else {;
    size_t found = _filename.find_last_of("/\\");
    string line_str, path_to_file = _filename.substr(0, found);
    while ( getline(myfile, line_str) )
      files.push_back(line_str);
  }
  return 1;
}


int main(int argc, char* argv[])
{
    // Read input parameters

    if ( argc != 5 )
    {
    help();
    exit(0);
    }

    // Parse the image paths

    vector<String> images_paths;
    getdir( argv[1], images_paths );

    Mat imgRef = imread(images_paths[0], IMREAD_COLOR);
    // Build intrinsics

    float f  = atof(argv[2]),
        cx = atof(argv[3]), cy = atof(argv[4]);

    Matx33d K = Matx33d( f, 0, cx,
                        0, f, cy,
                        0, 0,  1);

    /// Reconstruct the scene using the 2d images

    bool is_projective = true;
    bool refinement = true;
    int valRefinement = 0*SFM_REFINE_FOCAL_LENGTH;
    vector<Mat> Rs_est, ts_est, points3d_estimated;

    libmv_ReconstructionOptions optionsReconstruction(0, 1, valRefinement, 0, -1, 7000);
    libmv_CameraIntrinsicsOptions optionsModeleCamera(SFM_DISTORTION_MODEL_POLYNOMIAL,
        K(0, 0), K(1, 1), K(0, 2), K(1, 2), 0, 0, 0, 0, 0);
    Ptr<BaseSFM> reconstruction = SFMLibmvEuclideanReconstruction::create(optionsModeleCamera, optionsReconstruction);

    reconstruction->run(images_paths, K, Rs_est, ts_est, points3d_estimated);

    cout << "\n----------------------------\n" << endl;
    cout << "Optimized  SFM_REFINE_FOCAL_LENGTH" << endl;
    cout << "Reconstruction: " << endl;
    cout << "============================" << endl;
    cout << "Estimated 3D points: " << points3d_estimated.size() << endl;
    cout << "Estimated cameras: " << Rs_est.size() << endl;
    cout << "Refined intrinsics: " << endl << K << endl << endl;
    cout << "============================" << endl;

    optionsReconstruction.refine_intrinsics = 0*SFM_REFINE_PRINCIPAL_POINT;
    optionsReconstruction.nb_descriptors = 8000;
    optionsModeleCamera.focal_length_x = K(0, 0);
    optionsModeleCamera.focal_length_y = K(1, 1);
    optionsModeleCamera.principal_point_x = K(0, 2);
    optionsModeleCamera.principal_point_y = K(1, 2);
    reconstruction = SFMLibmvEuclideanReconstruction::create(optionsModeleCamera, optionsReconstruction);
    reconstruction->run(images_paths, K, Rs_est, ts_est, points3d_estimated);

    cout << "\n----------------------------\n" << endl;
    cout << "Optimized  SFM_REFINE_PRINCIPAL_POINT" << endl;
    cout << "Reconstruction: " << endl;
    cout << "============================" << endl;
    cout << "Estimated 3D points: " << points3d_estimated.size() << endl;
    cout << "Estimated cameras: " << Rs_est.size() << endl;
    cout << "Refined intrinsics: " << endl << K << endl << endl;
    cout << "============================" << endl;


    optionsReconstruction.refine_intrinsics = 0*(SFM_REFINE_FOCAL_LENGTH + SFM_REFINE_PRINCIPAL_POINT+SFM_REFINE_RADIAL_DISTORTION_K1 + SFM_REFINE_RADIAL_DISTORTION_K2);
    optionsReconstruction.nb_descriptors = 30000;
    optionsModeleCamera.focal_length_x = K(0, 0);
    optionsModeleCamera.focal_length_y = K(1, 1);
    optionsModeleCamera.principal_point_x = K(0, 2);
    optionsModeleCamera.principal_point_y = K(1, 2);
    reconstruction = SFMLibmvEuclideanReconstruction::create(optionsModeleCamera, optionsReconstruction);
    reconstruction->run(images_paths, K, Rs_est, ts_est, points3d_estimated);

    cout << "\n----------------------------\n" << endl;
    cout << "Optimized  0" << endl;
    cout << "Reconstruction: " << endl;
    cout << "============================" << endl;
    cout << "Estimated 3D points: " << points3d_estimated.size() << endl;
    cout << "Estimated cameras: " << Rs_est.size() << endl;
    cout << "Refined intrinsics: " << endl << K << endl << endl;
    cout << "============================" << endl;

    if(0 == 1)
    {
    // Print output
    cout << "\n----------------------------\n" << endl;
    cout << "Optimized  SFM_REFINE_FOCAL_LENGTH" << endl;
    cout << "Reconstruction: " << endl;
    cout << "============================" << endl;
    cout << "Estimated 3D points: " << points3d_estimated.size() << endl;
    cout << "Estimated cameras: " << Rs_est.size() << endl;
    cout << "Refined intrinsics: " << endl << K << endl << endl;
    cout << "============================" << endl;
    valRefinement = SFM_REFINE_PRINCIPAL_POINT | SFM_REFINE_FOCAL_LENGTH;
    reconstruct(images_paths, Rs_est, ts_est, K, points3d_estimated, is_projective, refinement, valRefinement);

    }



    // Print output

    cout << "\n----------------------------\n" << endl;
    cout << "Reconstruction: " << endl;
    cout << "Optimized  SFM_REFINE_PRINCIPAL_POINT" << endl;
    cout << "============================" << endl;
    cout << "Estimated 3D points: " << points3d_estimated.size() << endl;
    cout << "Estimated cameras: " << Rs_est.size() << endl;
    cout << "Refined intrinsics: " << endl << K << endl << endl;
    cout << "3D Visualization: " << endl;
    cout << "============================" << endl;


    /// Create 3D windows

    viz::Viz3d window("Coordinate Frame");
                window.setWindowSize(Size(500,500));
                window.setWindowPosition(Point(150,150));
                window.setBackgroundColor(); // black by default

    // Create the pointcloud
    cout << "Recovering points  ... ";

    // recover estimated points3d
    vector<Vec3f> point_cloud_est;
    for (int i = 0; i < points3d_estimated.size(); ++i)
        point_cloud_est.push_back(Vec3f(points3d_estimated[i]));

    cout << "[DONE]" << endl;


    /// Recovering cameras
    cout << "Recovering cameras ... ";

    vector<Affine3d> pathinv;

    /// Add the pointcloud
    vector<Vec3b> couleur;
    if ( point_cloud_est.size() > 0 )
    {
        vector<Mat> poseCamera(Rs_est.size());
        for (size_t i = 0; i < Rs_est.size(); ++i)
        {
            poseCamera[i] = Mat::eye(4, 4, CV_64F);
            Rs_est[i].copyTo(poseCamera[i](Range(0, 3), Range(0, 3)));
            ts_est[i].copyTo(poseCamera[i](Range(0, 3), Range(3, 4)));
            pathinv.push_back(Affine3d(Rs_est[i], ts_est[i]).inv());
        }
        cout << "[DONE]" << endl;

        for (int i = 0; i < point_cloud_est.size(); ++i)
        {
            Vec3f couleurPixel(0, 0, 0);
            Mat p1, p1a, p2, p3;
            int nbPixel = 0;
            convertPointsToHomogeneous(point_cloud_est[i].t(), p1);
            p1a = p1.reshape(1).t();
            p1a.convertTo(p1,CV_64F);
            for (int j = 0; j < pathinv.size(); j++)
            {
                p2 = poseCamera[j] * p1;
                p2.convertTo(p3, CV_64F);
                p3 = p3.rowRange(Range(0, 3));
                p2 = Mat(K) * p3;
                convertPointsFromHomogeneous(p2.t(), p3);
                Point pt(p3.at<double>(0, 0), p3.at<double>(0, 1));
                if (pt.x >= 0 && pt.x < imgRef.cols &&
                    pt.y >= 0 && pt.y < imgRef.rows)
                {
                    couleurPixel += imgRef.at<Vec3b>(pt);
                    nbPixel++;
                }
            }
            if (nbPixel == 0)
                couleur.push_back(viz::Color(viz::Color::red()));
            else
                couleur.push_back(couleurPixel / nbPixel);
        }
        cout << "Rendering points   ... ";
        cout << point_cloud_est.size() << endl;
        viz::WCloud cloud_widget(point_cloud_est, couleur);
        window.showWidget("point_cloud", cloud_widget);

        cout << "[DONE]" << endl;
    }
    else
    {
        cout << "Cannot render points: Empty pointcloud" << endl;
    }


    /// Add cameras
    if (pathinv.size() > 0 )
    {
        cout << "Rendering Cameras  ... ";

        window.showWidget("cameras_frames_and_lines", viz::WTrajectory(pathinv, viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
        window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(pathinv, K, 0.1, viz::Color::yellow()));

        window.setViewerPose(pathinv[0]);

        cout << "[DONE]" << endl;
    }
    else
    {
        cout << "Cannot render the cameras: Empty path" << endl;
    }

    /// Wait for key 'q' to close the window
    cout << endl << "Press 'q' to close each windows ... " << endl;

    window.spin();

    return 0;
}
