/*
 * =====================================================================================
 *
 *       Filename:  excluded_functions.cpp
 *
 *    Description:  Functions that the generator outputs incorrectly, either by them entirely
 *                  or by outputting them with incorrectly specified types.
 *
 *        Version:  1.0
 *        Created:  04/13/2014 12:06:39 AM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Arjun Comar 
 *
 * =====================================================================================
 */
#include <opencv2/c/excluded_functions.hpp>

extern "C" {

void cv_randu2(Mat* dst, Scalar* low, Scalar* high) {
	cv::randu(*dst, *low, *high);
}

void cv_goodFeaturesToTrack2(Mat* image, vector_Point2f* corners, int maxCorners, double qualityLevel, double minDistance, Mat* mask, int blockSize, bool useHarrisDetector, double k) {
	cv::goodFeaturesToTrack(*image, *corners, maxCorners, qualityLevel, minDistance, *mask, blockSize, useHarrisDetector, k);
}

Mat* cv_create_Mat_as_vectort(vector_Point2f* vec, bool copyData) {
    return new Mat(*vec, copyData);
}
	
Point* cv_RotatedRect_center(RotatedRect* self) {
    return new Point(self->center);
}
     
Size* cv_RotatedRect_size(RotatedRect* self) {
    return new Size(self->size);
}
     
RotatedRect* cv_create_RotatedRect(Point* center, Size* size, float angle) {
    return new RotatedRect(*center, *size, angle);
}

void cv_inRangeS(Mat* src, Scalar* lowerb, Scalar* upperb, Mat* dst) {
	cv::inRange(*src, *lowerb, *upperb, *dst);
}

int cv_createTrackbar(String* trackbarname, String* winname, int* value, int count, TrackbarCallback onChange, void* userdata) {
	return cv::createTrackbar(*trackbarname, *winname, value, count, onChange, userdata);
}

void cv_setMouseCallback(String* winname, MouseCallback onMouse, void* userdata) {
	return cv::setMouseCallback(*winname, onMouse, userdata);
}


}
