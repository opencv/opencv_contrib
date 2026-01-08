#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "opencv2/ximgproc.hpp"

using namespace std;
using namespace cv;

int main()
{
    Mat img = imread("opencv-logo.png", IMREAD_COLOR);
    resize(img, img, Size(), 0.5, 0.5, INTER_LINEAR_EXACT);

    /// Threshold the input image
    Mat img_grayscale, img_binary;
    cvtColor(img, img_grayscale,COLOR_BGR2GRAY);
    threshold(img_grayscale, img_binary, 0, 255, THRESH_OTSU | THRESH_BINARY_INV);

    /// Apply thinning to get a skeleton
    Mat img_thinning_ZS, img_thinning_GH;
    ximgproc::thinning(img_binary, img_thinning_ZS, ximgproc::THINNING_ZHANGSUEN);
    ximgproc::thinning(img_binary, img_thinning_GH, ximgproc::THINNING_GUOHALL);

    /// Make 3 channel images from thinning result
    Mat result_ZS(img.rows, img.cols, CV_8UC3), result_GH(img.rows, img.cols, CV_8UC3);

    Mat in[] = { img_thinning_ZS, img_thinning_ZS, img_thinning_ZS };
    Mat in2[] = { img_thinning_GH, img_thinning_GH, img_thinning_GH };
    int from_to[] = { 0,0, 1,1, 2,2 };
    mixChannels( in, 3, &result_ZS, 1, from_to, 3 );
    mixChannels( in2, 3, &result_GH, 1, from_to, 3 );

    /// Combine everything into a canvas
    Mat canvas(img.rows, img.cols * 3, CV_8UC3);
    img.copyTo( canvas( Rect(0, 0, img.cols, img.rows) ) );
    result_ZS.copyTo( canvas( Rect(img.cols, 0, img.cols, img.rows) ) );
    result_GH.copyTo( canvas( Rect(img.cols*2, 0, img.cols, img.rows) ) );

    /// Visualize result
    imshow("Skeleton", canvas); waitKey(0);

    return 0;
}
