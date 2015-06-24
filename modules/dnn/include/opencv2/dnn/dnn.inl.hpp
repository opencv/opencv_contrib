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
        CV_Assert(0 <= num && num < this->num() && 0 <= channel && channel < this->channels());
        return Mat(rows(), cols(), m.type(), this->rawPtr(num, channel));
    }

    inline int Blob::cols() const
    {
        CV_DbgAssert(m.dims > 2);
        return m.size[m.dims-1];
    }

    inline int Blob::rows() const
    {
        CV_DbgAssert(m.dims > 2);
        return m.size[m.dims-2];
    }

    inline Size Blob::size2() const
    {
        return Size(cols(), rows());
    }

    inline int Blob::channels() const
    {
        CV_DbgAssert(m.dims >= 3);
        return m.size[m.dims-3];
    }

    inline int Blob::num() const
    {
        CV_DbgAssert(m.dims == 4);
        return m.size[0];
    }

    inline Vec4i Blob::shape() const
    {
        CV_DbgAssert(m.dims == 4);
        return Vec4i(m.size.p);
    }

    inline int Blob::size(int index) const
    {
        CV_Assert(index >= 0 && index < dims());
        return sizes()[index];
    }

    inline size_t Blob::total(int startAxis, int endAxis) const
    {
        if (endAxis == -1)
            endAxis = dims();

        CV_Assert(0 <= startAxis && startAxis <= endAxis && endAxis <= dims());

        size_t size = 1; //assume that blob isn't empty
        for (int i = startAxis; i < endAxis; i++)
            size *= (size_t) sizes()[i];

        return size;
    }

    inline uchar* Blob::rawPtr(int num, int cn, int row, int col)
    {
        CV_DbgAssert(m.dims == 4);
        return m.data + num * m.step[0] + cn * m.step[1] + row * m.step[2] + col * m.step[3];
    }

    template<typename TFloat>
    TFloat *Blob::ptr(int n, int cn, int row, int col)
    {
        CV_Assert(m.type() == cv::DataType<TFloat>::type);
        CV_Assert(0 <= n && n < num() && 0 <= cn && cn < channels() && 0 <= row && row < rows() && 0 <= col && col < cols());
        return (TFloat*) rawPtr(n, cn, row, col);
    }

    inline int Blob::type() const
    {
        return m.depth();
    }

    inline bool Blob::isFloat() const
    {
        return (type() == CV_32F);
    }

    inline bool Blob::isDouble() const
    {
        return (type() == CV_32F);
    }

    inline const int * Blob::sizes() const
    {
        return &m.size[0];
    }

    inline int Blob::dims() const
    {
        return m.dims;
    }



}
}

#endif
