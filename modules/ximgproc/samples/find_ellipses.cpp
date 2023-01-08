// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

int main() {

    // load image
    Mat img = imread(samples::findFile("stuff.jpg"), IMREAD_COLOR);

    // check if image is loaded
    if (img.empty()) {
        std::cout << "fail to open image" << std::endl;
        return EXIT_FAILURE;
    }

    // create output array
    std::vector<Vec6f> ells;

    // test ellipse detection
    cv::ximgproc::findEllipses(img, ells, 0.4f, 0.7f, 0.02f);

    // print output
    for (unsigned i = 0; i < ells.size(); i++) {
        Vec6f ell = ells[i];
        std::cout << ell << std::endl;
        Scalar color(0, 0, 255);
        // draw ellipse on image
        ellipse(
                img,
                Point(cvRound(ell[0]), cvRound(ell[1])),
                Size(cvRound(ell[2]), cvRound(ell[3])),
                ell[5] * 180 / CV_PI, 0.0, 360.0, color, 3
        );
    }

    // show image
    imshow("result", img);
    waitKey();

    // end
    return 0;
}
