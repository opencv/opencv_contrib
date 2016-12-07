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

inline BlobShape::BlobShape()
{
    sz.allocate(4);
    for (size_t i = 0; i < sz.size(); i++)
        sz[i] = 1;
}

inline BlobShape BlobShape::all(int ndims, int fill)
{
    CV_Assert(ndims >= 0);
    BlobShape res;
    res.sz.allocate(ndims);
    for (int i = 0; i < ndims; i++)
        res.sz[i] = fill;
    return res;
}

inline BlobShape::BlobShape(int ndims, const int *sizes) : sz( (size_t)std::max(ndims, 0) )
{
    CV_Assert(ndims >= 0);
    if (!sizes)
        return;
    for (int i = 0; i < ndims; i++)
        sz[i] = sizes[i];
}

inline BlobShape::BlobShape(int s0) : sz(1)
{
    sz[0] = s0;
}

inline BlobShape::BlobShape(int s0, int s1) : sz(2)
{
    sz[0] = s0;
    sz[1] = s1;
}

inline BlobShape::BlobShape(int s0, int s1, int s2) : sz(3)
{
    sz[0] = s0;
    sz[1] = s1;
    sz[2] = s2;
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

inline int BlobShape::canonicalAxis(int axis) const
{
    CV_Assert(-dims() <= axis && axis < dims());
    return (axis < 0) ? axis + dims() : axis;
}

inline ptrdiff_t BlobShape::total() const
{
    if (dims() == 0)
        return 0;

    ptrdiff_t res = 1;
    for (int i = 0; i < dims(); i++)
        res *= sz[i];
    return res;
}

inline ptrdiff_t BlobShape::total(int startAxis, int endAxis) const
{
    if (isEmpty())
        return 0;

    if (endAxis == INT_MAX)
        endAxis = dims();
    else if (endAxis < 0)
        endAxis += dims();
    startAxis = (startAxis < 0) ? startAxis + dims() : startAxis;
    CV_Assert(0 <= startAxis && startAxis <= endAxis && endAxis <= dims());

    ptrdiff_t res = 1;
    for (int i = startAxis; i < endAxis; i++)
        res *= sz[i];
    return res;
}

inline BlobShape BlobShape::slice(int startAxis, int endAxis) const
{
    if (isEmpty())
        return BlobShape::empty();

    if (endAxis == INT_MAX)
        endAxis = dims();
    else if (endAxis < 0)
        endAxis += dims();
    startAxis = (startAxis < 0) ? startAxis + dims() : startAxis;
    CV_Assert(0 <= startAxis && startAxis <= endAxis && endAxis <= dims());

    BlobShape res(endAxis - startAxis, (const int*)NULL);
    for (int i = startAxis; i < endAxis; i++)
        res[i - startAxis] = sz[i];
    return res;
}

inline const int *BlobShape::ptr() const
{
    return sz;
}

inline int *BlobShape::ptr()
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

inline BlobShape BlobShape::like(const Mat &m)
{
    return BlobShape(m.dims, (const int*)m.size);
}

inline BlobShape BlobShape::like(const UMat &m)
{
    return BlobShape(m.dims, (const int*)m.size);
}

inline BlobShape BlobShape::empty()
{
    return BlobShape(0, (const int*)NULL);
}

inline bool BlobShape::isEmpty() const
{
    return dims() == 0;
}

inline BlobShape BlobShape::operator+(const BlobShape &r) const
{
    BlobShape newShape(this->dims() + r.dims(), (int*)NULL);
    for (int i = 0; i < this->dims(); i++)
        newShape[i] = (*this)[i];
    for (int i = 0; i < r.dims(); i++)
        newShape[this->dims() + i] = r[i];
    return newShape;
}

CV_EXPORTS std::ostream &operator<< (std::ostream &stream, const BlobShape &shape);

/////////////////////////////////////////////////////////////////////

#ifndef CV_DNN_UMAT
#   define CV_DNN_SWITCH_MU(cpu_expr, gpu_expr) (cpu_expr)
#else
#   define CV_DNN_SWITCH_MU(cpu_expr, gpu_expr) ((state == HEAD_AT_UMAT) ? (gpu_expr) : (cpu_expr))
#endif


inline int Blob::dims() const
{
    return CV_DNN_SWITCH_MU(m.dims, um.dims);
}

inline const int * Blob::sizes() const
{
    return CV_DNN_SWITCH_MU((const int*)m.size, (const int*)um.size);
}

inline int Blob::type() const
{
    return CV_DNN_SWITCH_MU(m.type(), um.type());
}

template<int n>
inline size_t Blob::offset(const Vec<int, n> &pos) const
{
    const MatStep &step = CV_DNN_SWITCH_MU(m.step, um.step);
    size_t ofs = 0;
    int i;
    for (i = 0; i < std::min(n, dims()); i++)
    {
        CV_DbgAssert(pos[i] >= 0 && pos[i] < size(i));
        ofs += step[i] * pos[i];
    }
    for (; i < dims(); i++)
        CV_DbgAssert(pos[i] == 0);
    CV_DbgAssert(ofs % elemSize() == 0);
    return ofs / elemSize();
}

inline int Blob::canonicalAxis(int axis) const
{
    CV_Assert(-dims() <= axis && axis < dims());
    return (axis < 0) ? axis + dims() : axis;
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

    size_t cnt = 1; //fix: assume that slice isn't empty
    for (int i = startAxis; i < endAxis; i++)
        cnt *= (size_t)sizes()[i];

    return cnt;
}

inline size_t Blob::offset(int n, int cn, int row, int col) const
{
    return offset(Vec4i(n, cn, row, col));
}

inline float *Blob::ptrf(int n, int cn, int row, int col)
{
    return matRef(false).ptr<float>() + offset(n, cn, row, col);
}

inline uchar *Blob::ptr(int n, int cn, int row, int col)
{
    Mat &mat = matRef(false);
    return mat.ptr() + mat.elemSize() * offset(n, cn, row, col);
}

template<typename Dtype>
inline Dtype* Blob::ptr(int n, int cn, int row, int col)
{
    CV_Assert(type() == cv::DataDepth<Dtype>::value);
    return (Dtype*) ptr(n, cn, row, col);
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

inline Mat& Blob::matRef(bool writeOnly)
{
#ifdef CV_DNN_UMAT
    updateMat(!writeOnly);
    state = HEAD_AT_MAT;
#else
    (void)writeOnly;
#endif
    return m;
}

inline const Mat& Blob::matRefConst() const
{
    CV_DNN_UMAT_ONLY( updateMat() );
    return m;
}

inline UMat &Blob::umatRef(bool writeOnly)
{
#ifndef CV_DNN_UMAT
    CV_Error(Error::GpuNotSupported, "");
    (void)writeOnly;
    return *(new UMat());
#else
    updateUMat(!writeOnly);
    state = HEAD_AT_UMAT;
    return um;
#endif
}

inline const UMat &Blob::umatRefConst() const
{
#ifndef CV_DNN_UMAT
    CV_Error(Error::GpuNotSupported, "");
    return *(new UMat());
#else
    updateUMat();
    return um;
#endif
}

template<>
inline Mat &Blob::getRef<Mat>(bool writeOnly)
{
    return matRef(writeOnly);
}

template<>
inline UMat &Blob::getRef<UMat>(bool writeOnly)
{
    return umatRef(writeOnly);
}

template<>
inline const Mat &Blob::getRefConst<Mat>() const
{
    return matRefConst();
}

template<>
inline const UMat &Blob::getRefConst<UMat>() const
{
    return umatRefConst();
}

inline Mat Blob::getPlane(int n, int cn)
{
    CV_Assert(dims() > 2);
    return Mat(dims() - 2, sizes() + 2, type(), ptr(n, cn));
}

inline Mat Blob::getPlanes(int n)
{
    CV_Assert(dims() > 3);
    return Mat(dims() - 1, sizes() + 1, type(), ptr(n));
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

inline Blob &Blob::shareFrom(const Blob &blob)
{
    this->m = blob.m;
#ifdef CV_DNN_UMAT
    this->um = blob.um;
    this->state = blob.state;
#endif
    return *this;
}

inline Blob &Blob::reshape(const BlobShape &newShape)
{
    if (!m.empty()) m = m.reshape(1, newShape.dims(), newShape.ptr());
#ifdef CV_DNN_UMAT
    if (!um.empty()) um = um.reshape(1, newShape.dims(), newShape.ptr());
#endif
    return *this;
}

inline Blob Blob::reshaped(const BlobShape &newShape) const
{
    Blob res(*this); //also, res.shareFrom(*this) could be used
    res.reshape(newShape);
    return res;
}

inline int Blob::elemSize() const
{
    return CV_ELEM_SIZE(type());
}

inline int Blob::getState() const
{
#ifdef CV_DNN_UMAT
    return this->state;
#else
    return m.empty() ? UNINITIALIZED : HEAD_AT_MAT;
#endif
}

}
}

#endif
