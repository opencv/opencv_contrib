/*
 * =====================================================================================
 *
 *       Filename:  scalar.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  03/17/14 18:49:28
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <opencv2/c/scalar.hpp>

extern "C" {

Scalar* cv_create_Scalar(double val0, double val1, double val2, double val3)
{
    return new Scalar(val0, val1, val2, val3);
}

Scalar* cv_create_scalarAll(double val0123)
{
    return new Scalar(Scalar::all(val0123));
}

}
