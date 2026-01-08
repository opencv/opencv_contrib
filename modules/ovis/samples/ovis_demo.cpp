#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/ovis.hpp>

#include <iostream>


#define KEY_ESCAPE 27

using namespace cv;

int main()
{
  Mat R;
  Vec3d t;

  const Size2i imsize(800, 600);
  const double focal_length = 800.0;

  //add some external resources
  ovis::addResourceLocation("packs/Sinbad.zip"); // shipped with Ogre

  //camera intrinsics
  Mat1d K = Mat1d::zeros(3, 3); // init empty matrix
  K.at<double>(0, 0) = focal_length; // f_x
  K.at<double>(1, 1) = focal_length; // f_y
  K.at<double>(0, 2) = 400; // t_x
  K.at<double>(1, 2) = 500; // t_y
  K.at<double>(2, 2) = 1; // f_z

  //observer scene
  Ptr<ovis::WindowScene> owin = ovis::createWindow(String("VR"), imsize);
  ovis::createGridMesh("ground", Size2i(10, 10), Size2i(10, 10));
  owin->createEntity("ground", "ground", Vec3f(1.57, 0, 0));
  owin->createCameraEntity("cam", K, imsize, 5);
  owin->createEntity("sinbad", "Sinbad.mesh", Vec3i(0, 0, 5), Vec3f(CV_PI/2.0, 0.0, 0.0)); // externally defined mesh
  owin->createLightEntity("sun", Vec3i(0, 0, -100));

  // setup and play idle animation
  owin->setEntityProperty("sinbad", ovis::EntityProperty::ENTITY_ANIMBLEND_MODE, Scalar(1)); // 1 = cumulative
  owin->playEntityAnimation("sinbad", "IdleBase");
  owin->playEntityAnimation("sinbad", "IdleTop");

  //interaction scene
  Ptr<ovis::WindowScene> iwin = ovis::createWindow(String("AR"), imsize, ovis::SCENE_SEPARATE | ovis::SCENE_INTERACTIVE);
  iwin->createEntity("sinbad", "Sinbad.mesh", Vec3i(0, -5, 0), Vec3f(CV_PI, 0.0, 0.0));
  iwin->createLightEntity("sun", Vec3i(0, 0, -100));
  iwin->setCameraIntrinsics(K, imsize);

  std::cout << "Press ESCAPE to exit demo" << std::endl;
  while (ovis::waitKey(1) != KEY_ESCAPE) {
      iwin->getCameraPose(R, t);
      owin->setEntityPose("cam", t, R);
  }

  return 1;
}
