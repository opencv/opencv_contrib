/*
 * =====================================================================================
 *
 *       Filename:  excluded_functions.hpp
 *
 *    Description:  Functions that the generator outputs incorrectly, either by them entirely
 *                  or by outputting them with incorrectly specified types.
 *
 *        Version:  1.0
 *        Created:  04/13/2014 12:00:46 AM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Arjun Comar
 *
 * =====================================================================================
 */

#include <opencv2/c/opencv_generated.hpp>

extern "C" {
void cv_randu2(Mat* dst, Scalar* low, Scalar* high);
void cv_goodFeaturesToTrack2(Mat* image, vector_Point2f* corners, int maxCorners, double qualityLevel, double minDistance, Mat* mask, int blockSize, bool useHarrisDetector, double k);
Mat* cv_create_Mat_as_vectort(vector_Point2f* vec, bool copyData);
Size* cv_RotatedRect_size(RotatedRect* self);
Point* cv_RotatedRect_center(RotatedRect* self);
RotatedRect* cv_create_RotatedRect(Point* center, Size* size, float angle);
void cv_inRangeS(Mat* src, Scalar* lowerb, Scalar* upperb, Mat* dst);
int cv_createTrackbar(String* trackbarname, String* winname, int* value, int count, TrackbarCallback onChange, void* userdata);
void cv_setMouseCallback(String* winname, MouseCallback onMouse, void* userdata);
}

