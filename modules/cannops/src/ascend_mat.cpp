// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <iostream>

namespace
{
class DefaultAllocator : public cv::cann::AscendMat::Allocator
{
public:
    std::shared_ptr<uchar> allocate(size_t size) CV_OVERRIDE;
    bool allocate(cv::cann::AscendMat* mat, int rows, int cols, size_t elemSize) CV_OVERRIDE;
};

std::shared_ptr<uchar> DefaultAllocator::allocate(size_t size)
{
    uchar* data;
    cv::cann::aclrtMallocWarpper((void**)(&data), size);
    return std::shared_ptr<uchar>(data, [](void* ptr) { cv::cann::aclrtFreeWarpper(ptr); });
}

bool DefaultAllocator::allocate(cv::cann::AscendMat* mat, int rows, int cols, size_t elemSize)
{
    size_t totalBytes = elemSize * cols * rows;

    // align by 32B.
    totalBytes = ((totalBytes + 32) & ~31);
    mat->data = allocate(totalBytes);
    mat->step = cols * elemSize;

    return true;
}

DefaultAllocator cannDefaultAllocator;
cv::cann::AscendMat::Allocator* g_defaultAllocator = &cannDefaultAllocator;
} // namespace

namespace cv
{
namespace cann
{
AscendMat::Allocator* AscendMat::defaultAllocator() { return g_defaultAllocator; }

void AscendMat::setDefaultAllocator(AscendMat::Allocator* allocator)
{
    CV_Assert(allocator != 0);
    g_defaultAllocator = allocator;
}

// TODO: this function is copied from matrix.cpp, which is a local symbol there and can not
// be refreneced, consider optimizing.
static int updateContinuityFlag(int flags, int dims, const int* size, const size_t* step)
{
    int i, j;
    for (i = 0; i < dims; i++)
    {
        if (size[i] > 1)
            break;
    }

    uint64 t = (uint64)size[std::min(i, dims - 1)] * CV_MAT_CN(flags);
    for (j = dims - 1; j > i; j--)
    {
        t *= size[j];
        if (step[j] * size[j] < step[j - 1])
            break;
    }

    if (j <= i && t == (uint64)(int)t)
        return flags | Mat::CONTINUOUS_FLAG;
    return flags & ~Mat::CONTINUOUS_FLAG;
}

void AscendMat::updateContinuityFlag()
{
    int sz[] = {rows, cols};
    size_t steps[] = {step, elemSize()};
    flags = cv::cann::updateContinuityFlag(flags, 2, sz, steps);
}

void AscendMat::create(int _rows, int _cols, int _type)
{
    CV_DbgAssert(_rows >= 0 && _cols >= 0);

    _type &= Mat::TYPE_MASK;

    if (rows == _rows && cols == _cols && type() == _type && data)
        return;

    if (_rows > 0 && _cols > 0)
    {
        flags = Mat::MAGIC_VAL + _type;
        rows = _rows;
        cols = _cols;

        const size_t esz = elemSize();

        bool allocSuccess = allocator->allocate(this, rows, cols, esz);

        if (!allocSuccess)
        {
            // custom allocator fails, try default allocator
            allocator = defaultAllocator();
            allocSuccess = allocator->allocate(this, rows, cols, esz);
            CV_Assert(allocSuccess);
        }

        if (esz * cols == step)
            flags |= Mat::CONTINUOUS_FLAG;

        datastart = data.get();
        dataend = data.get() + step * (rows - 1) + cols * esz;
    }
}

void AscendMat::upload(InputArray arr) { upload(arr, AscendStream::Null()); }

void AscendMat::upload(InputArray arr, AscendStream& stream)
{
    Mat mat = arr.getMat();
    CV_DbgAssert(!mat.empty());
    create(mat.rows, mat.cols, mat.type());
    aclrtMemcpy2dWarpper(data, 0, step, mat.data, mat.step[0], cols * elemSize(), rows, stream);
}

void AscendMat::download(OutputArray dst) const { download(dst, AscendStream::Null()); }

void AscendMat::download(OutputArray _dst, AscendStream& stream) const
{
    CV_DbgAssert(!empty());

    _dst.create(size(), type());
    Mat dst = _dst.getMat();
    aclrtMemcpy2dWarpper(dst.data, dst.step[0], data, 0, step, cols * elemSize(), rows, stream);
}

AscendMat::AscendMat(int rows_, int cols_, int type_, Scalar& s_, AscendMat::Allocator* allocator_)
    : flags(0), rows(rows_), cols(cols_), step(0), datastart(0), dataend(0), allocator(allocator_)
{
    create(rows_, cols_, type_);
    setTo(s_);
}

AscendMat::AscendMat(Size size_, int type_, Scalar& s_, AscendMat::Allocator* allocator_)
    : flags(0), rows(size_.height), cols(size_.width), step(0), datastart(0), dataend(0),
      allocator(allocator_)
{
    create(size_.height, size_.width, type_);
    setTo(s_);
}

AscendMat::AscendMat(InputArray _m, const Rect& roi) : AscendMat(_m, roi, AscendStream::Null()) {}

AscendMat::AscendMat(InputArray _m, const Rect& roi, AscendStream& stream)
    : rows(roi.height), cols(roi.width), allocator(defaultAllocator())
{
    AscendMat m;
    m.upload(_m, stream);
    step = m.step;
    data = m.data;
    flags = m.flags;
    CV_Assert(0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y &&
              0 <= roi.height && roi.y + roi.height <= m.rows);
    size_t esz = CV_ELEM_SIZE(flags);
    size_t sizeMem = esz * roi.width * roi.height * m.channels();
    size_t offset = roi.y * m.step + roi.x * esz;

    void* dst = malloc(sizeMem);
    size_t dpitch = roi.width * esz;
    std::shared_ptr<uchar> dstDevice = allocator->allocate(sizeMem);
    aclrtMemcpy2dWarpper(dst, dpitch, data, offset, step, dpitch, roi.height, stream);
    aclrtMemcpy2dWarpper(dstDevice, 0, dpitch, dst, dpitch, dpitch, roi.height, stream);
    data = dstDevice;
    step = dpitch;
    free(dst);
    updateContinuityFlag();
}

AscendMat& AscendMat::setTo(const Scalar& sc) { return setTo(sc, AscendStream::Null()); }

AscendMat& AscendMat::setTo(const Scalar& sc, AscendStream& stream)
{
    size_t totalBytes = (size_t)rows * cols * elemSize();
    if (totalBytes == 0)
        return *this;

    aclrtMemsetWarpper(data, 0, totalBytes, stream);
    AscendMat dst(rows, cols, type());
    arithm_op(*this, sc, dst, "Add", stream);
    swap(dst);

    return *this;
}

AscendMat& AscendMat::setTo(float sc) { return setTo(sc, AscendStream::Null()); }

AscendMat& AscendMat::setTo(float sc, AscendStream& stream)
{
    size_t totalBytes = (size_t)rows * cols * elemSize();
    if (totalBytes == 0)
        return *this;

    aclrtMemsetWarpper(data, 0, totalBytes, stream);

    AscendMat dst(rows, cols, type());
    arithm_op(*this, sc, dst, "Adds", stream);
    swap(dst);

    return *this;
}

void AscendMat::convertTo(AscendMat& dst, int rtype) const
{
    convertTo(dst, rtype, AscendStream::Null());
}

void AscendMat::convertTo(AscendMat& dst, int _rtype, AscendStream& stream) const
{
    int cn = channels();
    dst.create(rows, cols, CV_MAKE_TYPE(_rtype, cn));
    convertTo(dst, stream);
}

void AscendMat::convertTo(AscendMat& dst, AscendStream& stream) const
{
    OperatorRunner runner;
    runner.setOp("Cast")
        .addInput(*this, "x")
        .addOutput(dst, "y")
        .addAttr((int32_t)(getACLType(dst.depth())), "dst_type")
        .run(stream);
}
} // namespace cann
} // namespace cv
