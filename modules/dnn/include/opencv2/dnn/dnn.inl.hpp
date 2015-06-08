#ifndef __OPENCV_DNN_INL_HPP__
#define __OPENCV_DNN_INL_HPP__

#include <opencv2/dnn.hpp>

namespace cv
{
namespace dnn
{
    inline 
    Mat& Blob::getMatRef()
    {
        return m;
    }

    inline
    const Mat& Blob::getMatRef() const
    {
        return m;
    }

    inline
    Mat Blob::getMat()
    {
        return m;
    }


    Mat Blob::getMat(int num, int channel)
    {
        CV_Assert(false);
        return Mat();
    }
}
}

#endif
