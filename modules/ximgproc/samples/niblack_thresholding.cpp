/*
 * Sample C++ to demonstrate Niblack thresholding.
 *
 */

#include <iostream>
#include <cstdio>

#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/ximgproc.hpp"

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

Mat_<uchar> src, dst;

const int k_max_value = 10;
int k_from_slider = 0;
double k_actual = 0.0;

void on_trackbar(int, void*);

int main(int argc, char** argv)
{
    /*
     * Read filename from the command-line and load
     * corresponding gray-scale image.
     */
    if(argc != 2)
    {
        cout << "Usage: ./niblack_thresholding [IMAGE]\n";
        return 1;
    }
    const char* filename = argv[1];
    src = imread(filename, 1);

    namedWindow("k-slider", 1);
    string trackbar_name = "k";
    createTrackbar(trackbar_name, "k-slider", &k_from_slider, k_max_value, on_trackbar);
    on_trackbar(k_from_slider, 0);

    imshow("Source", src);
    waitKey(0);

    return 0;
}

void on_trackbar(int, void*)
{
    k_actual = (double)k_from_slider/k_max_value;
    niBlackThreshold(src, dst, 255, THRESH_BINARY, 3, k_actual);

    imshow("Destination", dst);
}
