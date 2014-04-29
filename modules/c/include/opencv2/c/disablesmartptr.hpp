/*
 * =====================================================================================
 *
 *       Filename:  smartptr.hpp
 *
 *    Description:  Header to disable smart pointers. This enables use of the C++ api from C
 *                  while leaving memory management up to the caller. This is good for
 *                  binding to other languages, but will cause problems if you expect
 *                  smart pointers to work as normal.
 *
 *        Version:  1.0
 *        Created:  04/24/2014 05:31:41 PM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Arjun Comar 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <opencv2/opencv.hpp>

using namespace cv;

