// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc/radon_transform.hpp>

using namespace cv;

int main() {
    Mat src = imread("peilin_plane.png", IMREAD_GRAYSCALE);
    Mat radon;
    ximgproc::RadonTransform(src, radon, 1, 0, 180, false, true);
    imshow("src image", src);
    imshow("Radon transform", radon);
    waitKey();
    return 0;
}
