/*
 * C++ sample to demonstrate Niblack thresholding.
 */

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

Mat_<uchar> src;
int k_ = 8;
int blockSize_ = 11;
int type_ = THRESH_BINARY;
int method_ = BINARIZATION_NIBLACK;

void on_trackbar(int, void*);

int main(int argc, char** argv)
{
    // read gray-scale image
    if(argc != 2)
    {
        cout << "Usage: ./niblack_thresholding [IMAGE]\n";
        return 1;
    }
    const char* filename = argv[1];
    src = imread(filename, IMREAD_GRAYSCALE);
    imshow("Source", src);

    namedWindow("Niblack", WINDOW_AUTOSIZE);
    createTrackbar("k", "Niblack", &k_, 20, on_trackbar);
    createTrackbar("blockSize", "Niblack", &blockSize_, 30, on_trackbar);
    createTrackbar("method", "Niblack", &method_, 3, on_trackbar);
    createTrackbar("threshType", "Niblack", &type_, 4, on_trackbar);
    on_trackbar(0, 0);
    waitKey(0);

    return 0;
}

void on_trackbar(int, void*)
{
    double k = static_cast<double>(k_-10)/10;                 // [-1.0, 1.0]
    int blockSize = 2*(blockSize_ >= 1 ? blockSize_ : 1) + 1; // 3,5,7,...,61
    int type = type_;  // THRESH_BINARY, THRESH_BINARY_INV,
                       // THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV
    int method = method_; //BINARIZATION_NIBLACK, BINARIZATION_SAUVOLA, BINARIZATION_WOLF, BINARIZATION_NICK
    Mat dst;
    niBlackThreshold(src, dst, 255, type, blockSize, k, method);
    imshow("Niblack", dst);
}
