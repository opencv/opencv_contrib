#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

using namespace cv;
using namespace cv::ximgproc;

#include <iostream>
using namespace std;

int sColor = 10, sSpace = 3,nbIter=4;

const char* window_name = "Rolling guidance filter";

void progressWindow(String windowName, Mat x)
{
    Mat y = Mat::zeros(x.size(), CV_8UC1);
    putText(y, "rollingGuidanceFilter in progress!", Point(10, x.rows / 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(255));
    imshow(windowName, y);
    waitKey(1);
}

/**
 * @function paillouFilter
 * @brief Trackbar callback
 */
static void rollingFilter(int, void *pm)
{
    Mat img = *((Mat*)pm);
    double sigmaColor(sColor/10.0), sigmaSpace(sSpace);
    Mat dst;
    progressWindow("rollingGuidanceFilter", img);
    rollingGuidanceFilter(img, dst, -1, sigmaColor, sigmaSpace, nbIter);
    imshow("rollingGuidanceFilter",dst );
}


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        cout << "usage: rollingGuidanceFilter_demo [image]" << endl;
        return 1;
    }
    Mat img = imread(argv[1]);
    if (img.empty())
    {
        cout << "File not found or empty image\n";
        return 1;
    }

    imshow("Original",img);
    Mat imgF;
    img.convertTo(imgF, CV_32F,1.0/255);
    namedWindow( window_name, WINDOW_KEEPRATIO);
    imshow(window_name, img);

    /// Create a Trackbar for user to enter threshold
    createTrackbar( "sColor",window_name, &sColor, 10, rollingFilter, &imgF );
    createTrackbar("sSpace", window_name, &sSpace, 400, rollingFilter, &imgF);
    createTrackbar("iter", window_name, &nbIter, 10, rollingFilter, &imgF);
    rollingFilter(0, &imgF);
    waitKey();
    return 0;
}
