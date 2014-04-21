/*
 * =====================================================================================
 *
 *       Filename:  scalar.hpp
 *
 *    Description:  Wrappers for the OpenCV Matrix class
 *
 *        Version:  1.0
 *        Created:  03/17/14 18:85:00
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Arjun Comar
 *
 * =====================================================================================
 */

#include <opencv2/c/opencv_generated.hpp>

extern "C" {
Scalar* cv_create_Scalar(double val0, double val1, double val2, double val3);
Scalar* cv_create_scalarAll(double val0123);
}
