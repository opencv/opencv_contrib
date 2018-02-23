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


void
usage()
{
  printf("usage: cvv_demo [-r WxH]\n");
  printf("-h       print this help\n");
  printf("-r WxH   change resolution to width W and height H\n");
}


int
main(int argc, char** argv)
{
  cv::Size* resolution = nullptr;

  // parser keys
  const char *keys =
      "{ help h usage ? |    | show this message }"
      "{ resolution r   |0x0| resolution to width and height in the format WxH }";
  CommandLineParser parser(argc, argv, keys);
  string res(parser.get<string>("resolution"));
  if (parser.has("help")) {
    usage();
    return 0;
  }
  if (res != "0x0") {
    char dummych;
    resolution = new cv::Size();
    if (sscanf(res.c_str(), "%d%c%d", &resolution->width, &dummych, &resolution->height) != 3) {
      cout << res << " not a valid resolution" << endl;
      return 1;
    }
  }

  // setup video capture
  cv::VideoCapture capture(0);
  if (!capture.isOpened()) {
    std::cout << "Could not open VideoCapture" << std::endl;
    return 3;
  }

  if (resolution) {
    printf("Setting resolution to %dx%d\n", resolution->width, resolution->height);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, resolution->width);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, resolution->height);
    delete resolution;
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
    cv::cvtColor(imgRead, imgGray, CV_BGR2GRAY);
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
