// system includes
#include <iostream>

// library includes
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>

#define CVVISUAL_DEBUGMODE
#include <opencv2/cvv/debug_mode.hpp>
#include <opencv2/cvv/show_image.hpp>
#include <opencv2/cvv/filter.hpp>
#include <opencv2/cvv/dmatch.hpp>
#include <opencv2/cvv/final_show.hpp>

using namespace std;
using namespace cv;

template<class T> std::string toString(const T& p_arg)
{
  std::stringstream ss;

  ss << p_arg;

  return ss.str();
}




int
main(int argc, char** argv)
{
  cv::Size* resolution = nullptr;

  // parser keys
  const char *keys =
      "{ help h usage ?  |   | show this message }"
      "{ width W         |  0| camera resolution width. leave at 0 to use defaults }"
      "{ height H        |  0| camera resolution height. leave at 0 to use defaults }";

  CommandLineParser parser(argc, argv, keys);
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  int res_w = parser.get<int>("width");
  int res_h = parser.get<int>("height");

  // setup video capture
  cv::VideoCapture capture(0);
  if (!capture.isOpened()) {
    std::cout << "Could not open VideoCapture" << std::endl;
    return 1;
  }

  if (res_w>0 && res_h>0) {
    printf("Setting resolution to %dx%d\n", res_w, res_h);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, res_w);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, res_h);
  }


  cv::Mat prevImgGray;
  std::vector<cv::KeyPoint> prevKeypoints;
  cv::Mat prevDescriptors;

  int maxFeatureCount = 500;
  Ptr<ORB> detector = ORB::create(maxFeatureCount);

  cv::BFMatcher matcher(cv::NORM_HAMMING);

  for (int imgId = 0; imgId < 10; imgId++) {
    // capture a frame
    cv::Mat imgRead;
    capture >> imgRead;
    printf("%d: image captured\n", imgId);

    std::string imgIdString{"imgRead"};
    imgIdString += toString(imgId);
		cvv::showImage(imgRead, CVVISUAL_LOCATION, imgIdString.c_str());

    // convert to grayscale
    cv::Mat imgGray;
    cv::cvtColor(imgRead, imgGray, COLOR_BGR2GRAY);
		cvv::debugFilter(imgRead, imgGray, CVVISUAL_LOCATION, "to gray");

    // detect ORB features
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(imgGray, cv::noArray(), keypoints, descriptors);
    printf("%d: detected %zd keypoints\n", imgId, keypoints.size());

    // match them to previous image (if available)
    if (!prevImgGray.empty()) {
      std::vector<cv::DMatch> matches;
      matcher.match(prevDescriptors, descriptors, matches);
      printf("%d: all matches size=%zd\n", imgId, matches.size());
      std::string allMatchIdString{"all matches "};
      allMatchIdString += toString(imgId-1) + "<->" + toString(imgId);
      cvv::debugDMatch(prevImgGray, prevKeypoints, imgGray, keypoints, matches, CVVISUAL_LOCATION, allMatchIdString.c_str());

      // remove worst (as defined by match distance) bestRatio quantile
      double bestRatio = 0.8;
      std::sort(matches.begin(), matches.end());
      matches.resize(int(bestRatio * matches.size()));
      printf("%d: best matches size=%zd\n", imgId, matches.size());
      std::string bestMatchIdString{"best " + toString(bestRatio) + " matches "};
      bestMatchIdString += toString(imgId-1) + "<->" + toString(imgId);
      cvv::debugDMatch(prevImgGray, prevKeypoints, imgGray, keypoints, matches, CVVISUAL_LOCATION, bestMatchIdString.c_str());
    }

    prevImgGray = imgGray;
    prevKeypoints = keypoints;
    prevDescriptors = descriptors;
  }

  cvv::finalShow();

  return 0;
}
