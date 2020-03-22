#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/ovis.hpp>
#include <opencv2/aruco.hpp>

#include <iostream>


#define KEY_ESCAPE 27

using namespace cv;

int main()
{
  Mat img;
  std::vector<std::vector<Point2f>> corners;
  std::vector<int> ids;
  std::vector<Vec3d> rvecs;
  std::vector<Vec3d> tvecs;

  const Size2i imsize(800, 600);
  const double focal_length = 800.0;

  // aruco
  Ptr<aruco::Dictionary> adict = aruco::getPredefinedDictionary(aruco::DICT_4X4_50);
  //Mat out_img;
  //aruco::drawMarker(adict, 0, 400, out_img);
  //imshow("marker", out_img);

  // random calibration data, your mileage may vary
  Mat1d cm = Mat1d::zeros(3, 3); // init empty matrix
  cm.at<double>(0, 0) = focal_length; // f_x
  cm.at<double>(1, 1) = focal_length; // f_y
  cm.at<double>(2, 2) = 1; // f_z
  Mat K = getDefaultNewCameraMatrix(cm, imsize, true);

  // AR scene
  ovis::addResourceLocation("packs/Sinbad.zip"); // shipped with Ogre

  Ptr<ovis::WindowScene> win = ovis::createWindow(String("arucoAR"), imsize, ovis::SCENE_INTERACTIVE | ovis::SCENE_AA);
  win->setCameraIntrinsics(K, imsize);
  win->createEntity("sinbad", "Sinbad.mesh", Vec3i(0, 0, 5), Vec3f(1.57, 0.0, 0.0));
  win->createLightEntity("sun", Vec3i(0, 0, 100));

  // video capture
  VideoCapture cap{0};
  cap.set(CAP_PROP_FRAME_WIDTH, imsize.width);
  cap.set(CAP_PROP_FRAME_HEIGHT, imsize.height);

  std::cout << "Press ESCAPE to exit demo" << std::endl;
  while (ovis::waitKey(1) != KEY_ESCAPE) {
    cap.read(img);
    win->setBackground(img);
    aruco::detectMarkers(img, adict, corners, ids);

    waitKey(1);

    if (ids.size() == 0)
      continue;

    aruco::estimatePoseSingleMarkers(corners, 5, K, noArray(), rvecs, tvecs);
    win->setCameraPose(tvecs.at(0), rvecs.at(0), true);
  }

  return 0;
}
