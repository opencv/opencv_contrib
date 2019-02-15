/*
 * shape_context.cpp -- Shape context demo for shape matching
 */
#include <iostream>
#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_SHAPE

#include "opencv2/shape.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/core/utility.hpp"
#include <string>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

static void help()
{
    printf("\nThis program demonstrates how to use common interface for shape transformers\n"
           "Call\n"
           "shape_transformation [image1] [image2]\n");
}

int main(int argc, char** argv)
{
    help();
    if (argc < 3)
    {
      printf("Not enough parameters\n");
      return -1;
    }
    Mat img1 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat img2 = imread(argv[2], IMREAD_GRAYSCALE);
    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }

    // detecting keypoints & computing descriptors
    Ptr<SURF> surf = SURF::create(5000);
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    surf->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
    surf->detectAndCompute(img2, Mat(), keypoints2, descriptors2);

    // matching descriptors
    BFMatcher matcher(surf->defaultNorm());
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // drawing the results
    namedWindow("matches", 1);
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    imshow("matches", img_matches);

    // extract points
    vector<Point2f> pts1, pts2;
    for (size_t ii=0; ii<keypoints1.size(); ii++)
        pts1.push_back( keypoints1[ii].pt );
    for (size_t ii=0; ii<keypoints2.size(); ii++)
        pts2.push_back( keypoints2[ii].pt );

    // Apply TPS
    Ptr<ThinPlateSplineShapeTransformer> mytps = createThinPlateSplineShapeTransformer(25000); //TPS with a relaxed constraint
    mytps->estimateTransformation(pts1, pts2, matches);
    mytps->warpImage(img2, img2);

    imshow("Tranformed", img2);
    waitKey(0);

    return 0;
}

#else

int main()
{
    std::cerr << "OpenCV was built without shape module" << std::endl;
    return 0;
}

#endif // HAVE_OPENCV_SHAPE
