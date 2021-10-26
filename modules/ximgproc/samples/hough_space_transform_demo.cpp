// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc/hough_space_transform.hpp>

int main() {
	cv::Mat src = cv::imread("peilin_plane.png", cv::IMREAD_GRAYSCALE);
	cv::Mat hough;
	cv::ximgproc::HoughSpaceTransform(src, hough, 1, 0, 180, false, true);
	cv::imshow("src image", src);
	cv::imshow("hough space", hough);
	cv::waitKey();
	return 0;
}