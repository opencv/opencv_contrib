/*
 * =====================================================================================
 *
 *       Filename:  point.hpp
 *
 *    Description:  Wrapper header for the OpenCV Point class(es)
 *
 *        Version:  1.0
 *        Created:  10/02/2013 11:54:37 AM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Arjun Comar 
 *
 * =====================================================================================
 */

#include <opencv2/c/opencv_generated.hpp>

#define ADD_POINT_FUNC_HEADERS(t, tn) \
    Point2##t * cv_create_Point2##t ( tn x,  tn y); \
    Point3##t * cv_create_Point3##t ( tn x,  tn y,  tn z); \
    tn cv_Point2##t##_getX( Point2##t * self); \
    tn cv_Point2##t##_getY( Point2##t * self); \
    tn cv_Point3##t##_getX( Point3##t * self); \
    tn cv_Point3##t##_getY( Point3##t * self); \
    tn cv_Point3##t##_getZ( Point3##t * self); \
    tn cv_Point2##t##_dot( Point2##t * self, Point2##t * other); \
    tn cv_Point3##t##_dot( Point3##t * self, Point3##t * other); \
    Point3##t * cv_Point3##t##_cross(Point3##t * self, Point3##t * other);

extern "C" {
    ADD_POINT_FUNC_HEADERS(i, int);
    ADD_POINT_FUNC_HEADERS(f, float);
    ADD_POINT_FUNC_HEADERS(d, double);
}

