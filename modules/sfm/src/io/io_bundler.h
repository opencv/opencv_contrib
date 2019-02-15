/*
 Based on TheiaSfM library.
 https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/io/read_bundler_files.cc

 Adapted by Edgar Riba <edgar.riba@gmail.com>

*/

#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>

// The bundle files contain the estimated scene and camera geometry have the
// following format:
//     # Bundle file v0.3
//     <num_cameras> <num_points>   [two integers]
//     <camera1>
//     <camera2>
//        ...
//     <cameraN>
//     <point1>
//     <point2>
//        ...
//     <pointM>
// Each camera entry <cameraI> contains the estimated camera intrinsics and
// extrinsics, and has the form:
//     <f> <k1> <k2>   [the focal length, followed by two radial distortion
//                      coeffs]
//     <R>             [a 3x3 matrix representing the camera rotation]
//     <t>             [a 3-vector describing the camera translation]
// The cameras are specified in the order they appear in the list of images.
//
// Each point entry has the form:
//     <position>      [a 3-vector describing the 3D position of the point]
//     <color>         [a 3-vector describing the RGB color of the point]
//     <view list>     [a list of views the point is visible in]
//
// The view list begins with the length of the list (i.e., the number of cameras
// the point is visible in). The list is then given as a list of quadruplets
// <camera> <key> <x> <y>, where <camera> is a camera index, <key> the index of
// the SIFT keypoint where the point was detected in that camera, and <x> and
// <y> are the detected positions of that keypoint. Both indices are 0-based
// (e.g., if camera 0 appears in the list, this corresponds to the first camera
// in the scene file and the first image in "list.txt"). The pixel positions are
// floating point numbers in a coordinate system where the origin is the center
// of the image, the x-axis increases to the right, and the y-axis increases
// towards the top of the image. Thus, (-w/2, -h/2) is the lower-left corner of
// the image, and (w/2, h/2) is the top-right corner (where w and h are the
// width and height of the image).
static bool readBundlerFile(const std::string &file,
                     std::vector<cv::Matx33d> &Rs,
                     std::vector<cv::Vec3d> &Ts,
                     std::vector<cv::Matx33d> &Ks,
                     std::vector<cv::Vec3d> &points3d) {

  // Read in num cameras, num points.
  std::ifstream ifs(file.c_str(), std::ios::in);
  if (!ifs.is_open()) {
    std::cout << "Cannot read the file from " << file << std::endl;
    return false;
  }

  const cv::Matx33d bundler_to_opencv(1, 0, 0, 0, -1, 0, 0, 0, -1);

  std::string header_string;
  std::getline(ifs, header_string);

  // If the first line starts with '#' then it is a comment, so skip it!
  if (header_string[0] == '#') {
    std::getline(ifs, header_string);
  }
  const char* p = header_string.c_str();
  char* p2;
  const int num_cameras = strtol(p, &p2, 10);

  p = p2;
  const int num_points = strtol(p, &p2, 10);

  // Read in the camera params.
  for (int i = 0; i < num_cameras; i++) {
    // Read in focal length, radial distortion.
    std::string internal_params;
    std::getline(ifs, internal_params);
    p = internal_params.c_str();
    const double focal_length = strtod(p, &p2);
    p = p2;
    //const double k1 = strtod(p, &p2);
    p = p2;
    //const double k2 = strtod(p, &p2);
    p = p2;

    cv::Matx33d intrinsics;
    intrinsics(0,0) = intrinsics(1,1) = focal_length;
    Ks.push_back(intrinsics);

    // Read in rotation (row-major).
    cv::Matx33d rotation;
    for (int r = 0; r < 3; r++) {
      std::string rotation_row;
      std::getline(ifs, rotation_row);
      p = rotation_row.c_str();

      for (int c = 0; c < 3; c++) {
        rotation(r, c) = strtod(p, &p2);
        p = p2;
      }
    }

    std::string translation_string;
    std::getline(ifs, translation_string);
    p = translation_string.c_str();
    cv::Vec3d translation;
    for (int j = 0; j < 3; j++) {
      translation(j) = strtod(p, &p2);
      p = p2;
    }

    rotation = bundler_to_opencv * rotation;
    translation =  bundler_to_opencv * translation;

    cv::Matx33d rotation_t = rotation.t();
    translation = -1.0 * rotation_t * translation;

    Rs.push_back(rotation);
    Ts.push_back(translation);

    if ((i + 1) % 100 == 0 || i == num_cameras - 1) {
      std::cout << "\r Loading parameters for camera " << i + 1 << " / "
                << num_cameras << std::flush;
    }
  }
  std::cout << std::endl;

  // Read in each 3D point and correspondences.
  for (int i = 0; i < num_points; i++) {
    // Read position.
    std::string position_str;
    std::getline(ifs, position_str);
    p = position_str.c_str();
    cv::Vec3d position;
    for (int j = 0; j < 3; j++) {
      position(j) = strtod(p, &p2);
      p = p2;
    }
    points3d.push_back(position);

    // Read color.
    std::string color_str;
    std::getline(ifs, color_str);
    p = color_str.c_str();
    cv::Vec3d color;
    for (int j = 0; j < 3; j++) {
      color(j) = static_cast<double>(strtol(p, &p2, 10)) / 255.0;
      p = p2;
    }

    // Read viewlist.
    std::string view_list_string;
    std::getline(ifs, view_list_string);
    p = view_list_string.c_str();
    const int num_views = strtol(p, &p2, 10);
    p = p2;

    // Reserve the view list for this 3D point.
    for (int j = 0; j < num_views; j++) {
      // Camera key x y
      //const int camera_index = strtol(p, &p2, 10);
      p = p2;
      // Returns the index of the sift descriptor in the camera for this track.
      strtol(p, &p2, 10);
      p = p2;
      //const float x_pos = strtof(p, &p2);
      p = p2;
      //const float y_pos = strtof(p, &p2);
      p = p2;

    }

    if ((i + 1) % 100 == 0 || i == num_points - 1) {
      std::cout << "\r Loading 3D points " << i + 1 << " / " << num_points
                << std::flush;
    }
  }

  std::cout << std::endl;
  ifs.close();

  return true;
}
