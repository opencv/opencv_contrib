// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANNOPS_CANN_INL_HPP
#define OPENCV_CANNOPS_CANN_INL_HPP

#include "opencv2/cann.hpp"

namespace cv
{
namespace cann
{
inline AscendMat::AscendMat(AscendMat::Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), datastart(0), dataend(0),
      allocator(allocator_)
{
    // Empty mat is also continuous.
    flags |= Mat::CONTINUOUS_FLAG;
}

inline AscendMat::AscendMat(int rows_, int cols_, int type_, AscendMat::Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), datastart(0), dataend(0),
      allocator(allocator_)
{
    if (rows_ > 0 && cols_ > 0)
        create(rows_, cols_, type_);
}

inline AscendMat::AscendMat(Size size_, int type_, AscendMat::Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), datastart(0), dataend(0),
      allocator(allocator_)
{
    if (size_.height > 0 && size_.width > 0)
        create(size_.height, size_.width, type_);
}

inline AscendMat::AscendMat(InputArray arr, AscendStream& stream, AscendMat::Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), datastart(0), dataend(0),
      allocator(allocator_)
{
    upload(arr, stream);
}

inline AscendMat::AscendMat(const AscendMat& m)
    : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data),
      datastart(m.datastart), dataend(m.dataend), allocator(m.allocator)
{}

inline AscendMat& AscendMat::operator=(const AscendMat& m)
{
    if (this != &m)
    {
        AscendMat temp(m);
        swap(temp);
    }

    return *this;
}

inline void AscendMat::swap(AscendMat& b)
{
    std::swap(flags, b.flags);
    std::swap(rows, b.rows);
    std::swap(cols, b.cols);
    std::swap(step, b.step);
    std::swap(data, b.data);
    std::swap(datastart, b.datastart);
    std::swap(dataend, b.dataend);
    std::swap(allocator, b.allocator);
}

inline bool AscendMat::isContinuous() const { return (flags & Mat::CONTINUOUS_FLAG) != 0; }

inline size_t AscendMat::elemSize() const { return CV_ELEM_SIZE(flags); }

inline size_t AscendMat::elemSize1() const { return CV_ELEM_SIZE1(flags); }

inline int AscendMat::type() const { return CV_MAT_TYPE(flags); }

inline int AscendMat::depth() const { return CV_MAT_DEPTH(flags); }

inline int AscendMat::channels() const { return CV_MAT_CN(flags); }

inline size_t AscendMat::step1() const { return step / elemSize1(); }

inline Size AscendMat::size() const { return Size(cols, rows); }

inline bool AscendMat::empty() const { return data == 0; }

inline AscendStream::AscendStream(const Ptr<AscendStream::Impl>& impl) : impl_(impl) {}

inline AscendEvent::AscendEvent(const Ptr<AscendEvent::Impl>& impl) : impl_(impl) {}
} // namespace cann
} // namespace cv

#endif // OPENCV_CANNOPS_CANN_INL_HPP
