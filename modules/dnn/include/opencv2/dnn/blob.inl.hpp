/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_DNN_DNN_BLOB_INL_HPP__
#define __OPENCV_DNN_DNN_BLOB_INL_HPP__
#include "blob.hpp"

namespace cv
{
namespace dnn
{

inline BlobShape::BlobShape(int ndims, int fill) : sz( (size_t)std::max(ndims, 0) )
{
    CV_Assert(ndims >= 0);
    for (int i = 0; i < ndims; i++)
        sz[i] = fill;
}

inline BlobShape::BlobShape(int ndims, const int *sizes) : sz( (size_t)std::max(ndims, 0) )
{
    CV_Assert(ndims >= 0);
    for (int i = 0; i < ndims; i++)
        sz[i] = sizes[i];
}

inline BlobShape::BlobShape(int num, int cn, int rows, int cols) : sz(4)
{
    sz[0] = num;
    sz[1] = cn;
    sz[2] = rows;
    sz[3] = cols;
}

inline BlobShape::BlobShape(const std::vector<int> &sizes) : sz( sizes.size() )
{
    for (int i = 0; i < (int)sizes.size(); i++)
        sz[i] = sizes[i];
}

template<int n>
inline BlobShape::BlobShape(const Vec<int, n> &shape) : sz(n)
{
    for (int i = 0; i < n; i++)
        sz[i] = shape[i];
}

inline int BlobShape::dims() const
{
    return (int)sz.size();
}

inline int BlobShape::xsize(int axis) const
{
    if (axis < -dims() || axis >= dims())
        return 1;

    return sz[(axis < 0) ? axis + dims() : axis];
}

inline int BlobShape::size(int axis) const
{
    CV_Assert(-dims() <= axis && axis < dims());
    return sz[(axis < 0) ? axis + dims() : axis];
}

inline int &BlobShape::size(int axis)
{
    CV_Assert(-dims() <= axis && axis < dims());
    return sz[(axis < 0) ? axis + dims() : axis];
}

inline int BlobShape::operator[] (int axis) const
{
    CV_Assert(-dims() <= axis && axis < dims());
    return sz[(axis < 0) ? axis + dims() : axis];
}

inline int &BlobShape::operator[] (int axis)
{
    CV_Assert(-dims() <= axis && axis < dims());
    return sz[(axis < 0) ? axis + dims() : axis];
}

inline ptrdiff_t BlobShape::total()
{
    if (dims() == 0)
        return 0;

    ptrdiff_t res = 1;
    for (int i = 0; i < dims(); i++)
        res *= sz[i];
    return res;
}

inline const int *BlobShape::ptr() const
{
    return sz;
}

inline bool BlobShape::equal(const BlobShape &other) const
{
    if (this->dims() != other.dims())
        return false;

    for (int i = 0; i < other.dims(); i++)
    {
        if (sz[i] != other.sz[i])
            return false;
    }

    return true;
}

inline bool BlobShape::operator==(const BlobShape &r) const
{
    return this->equal(r);
}

CV_EXPORTS std::ostream &operator<< (std::ostream &stream, const BlobShape &shape);

/////////////////////////////////////////////////////////////////////

inline int Blob::canonicalAxis(int axis) const
{
    CV_Assert(-dims() <= axis && axis < dims());
    return (axis < 0) ? axis + dims() : axis;
}

inline int Blob::dims() const
{
    return m.dims;
}

inline int Blob::xsize(int axis) const
{
    if (axis < -dims() || axis >= dims())
        return 1;

    return sizes()[(axis < 0) ? axis + dims() : axis];
}

inline int Blob::size(int axis) const
{
    CV_Assert(-dims() <= axis && axis < dims());
    return sizes()[(axis < 0) ? axis + dims() : axis];
}

inline size_t Blob::total(int startAxis, int endAxis) const
{
    if (startAxis < 0)
        startAxis += dims();

    if (endAxis == INT_MAX)
        endAxis = dims();
    else if (endAxis < 0)
        endAxis += dims();

    CV_Assert(0 <= startAxis && startAxis <= endAxis && endAxis <= dims());

    size_t size = 1; //fix: assume that slice isn't empty
    for (int i = startAxis; i < endAxis; i++)
        size *= (size_t)sizes()[i];

    return size;
}


template<int n>
inline size_t Blob::offset(const Vec<int, n> &pos) const
{
    size_t ofs = 0;
    int i;
    for (i = 0; i < std::min(n, dims()); i++)
    {
        CV_DbgAssert(pos[i] >= 0 && pos[i] < size(i));
        ofs = ofs * (size_t)size(i) + pos[i];
    }
    for (; i < dims(); i++)
        ofs *= (size_t)size(i);
    return ofs;
}

inline size_t Blob::offset(int n, int cn, int row, int col) const
{
    return offset(Vec4i(n, cn, row, col));
}

inline float *Blob::ptrf(int n, int cn, int row, int col)
{
    CV_Assert(type() == CV_32F);
    return (float*)m.data + offset(n, cn, row, col);
}

inline uchar *Blob::ptr(int n, int cn, int row, int col)
{
    return m.data + m.elemSize() * offset(n, cn, row, col);
}

template<typename TFloat>
inline TFloat* Blob::ptr(int n, int cn, int row, int col)
{
    CV_Assert(type() == cv::DataDepth<TFloat>::value);
    return (TFloat*) ptr(n, cn, row, col);
}

inline BlobShape Blob::shape() const
{
    return BlobShape(dims(), sizes());
}

inline bool Blob::equalShape(const Blob &other) const
{
    if (this->dims() != other.dims())
        return false;

    for (int i = 0; i < dims(); i++)
    {
        if (this->sizes()[i] != other.sizes()[i])
            return false;
    }
    return true;
}

inline Mat& Blob::matRef()
{
    return m;
}

inline const Mat& Blob::matRefConst() const
{
    return m;
}

inline UMat &Blob::umatRef()
{
    CV_Error(Error::StsNotImplemented, "");
    return *(new UMat());
}

inline const UMat &Blob::umatRefConst() const
{
    CV_Error(Error::StsNotImplemented, "");
    return *(new UMat());
}

inline Mat Blob::getPlane(int n, int cn)
{
    CV_Assert(dims() > 2);
    return Mat(dims() - 2, sizes() + 2, type(), ptr(n, cn));
}

inline int Blob::cols() const
{
    return xsize(3);
}

inline int Blob::rows() const
{
    return xsize(2);
}

inline int Blob::channels() const
{
    return xsize(1);
}

inline int Blob::num() const
{
    return xsize(0);
}

inline Size Blob::size2() const
{
    return Size(cols(), rows());
}

inline int Blob::type() const
{
    return m.depth();
}

inline const int * Blob::sizes() const
{
    return &m.size[0];
}


inline Blob &Blob::shareFrom(const Blob &blob)
{
    this->m = blob.m;
    return *this;
}

inline Blob &Blob::reshape(const BlobShape &shape)
{
    m = m.reshape(1, shape.dims(), shape.ptr());
    return *this;
}

}
}

#endif
