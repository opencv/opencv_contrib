#ifndef __OPENCV_DNN_INL_HPP__
#define __OPENCV_DNN_INL_HPP__

#include <opencv2/dnn.hpp>

namespace cv
{
namespace dnn
{
    inline Mat& Blob::getMatRef()
    {
        return m;
    }

    inline const Mat& Blob::getMatRef() const
    {
        return m;
    }

    inline Mat Blob::getMat()
    {
        return m;
    }


    inline Mat Blob::getMat(int num, int channel)
    {
        CV_Assert(false);
        return Mat();
    }

    inline
    int Blob::cols() const
    {
        CV_DbgAssert(m.dims > 2);
        return m.size[m.dims-1];
    }

    inline
    int Blob::rows() const
    {
        CV_DbgAssert(m.dims > 2);
        return m.size[m.dims-2];
    }

    inline
    Size Blob::size() const
    {
        return Size(cols(), rows());
    }
    
    inline
    int Blob::channels() const
    {
        CV_DbgAssert(m.dims >= 3);
        return m.size[m.dims-3];
    }
    
    inline
    int Blob::num() const
    {
        CV_DbgAssert(m.dims == 4);
        return m.size[0];
    }

    inline
    Vec4i Blob::shape() const
    {
        CV_DbgAssert(m.dims == 4);
        return Vec4i(m.size.p);
    }
}
}

#endif
