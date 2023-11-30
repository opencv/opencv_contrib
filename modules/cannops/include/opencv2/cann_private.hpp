// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANNOPS_CANN_PRIVATE_HPP
#define OPENCV_CANNOPS_CANN_PRIVATE_HPP
#include "opencv2/cann.hpp"

namespace cv
{
namespace cann
{
void arithm_op(const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const char* op,
               AscendStream& stream);
void arithm_op(const AscendMat& src, const Scalar& sc, AscendMat& dst, const char* op,
               AscendStream& stream);
void arithm_op(const Scalar& sc, const AscendMat& src, AscendMat& dst, const char* op,
               AscendStream& stream);
void arithm_op(const AscendMat& src, AscendMat& dst, const char* op, AscendStream& stream);
void arithm_op(const AscendMat& src, float scalar, AscendMat& dst, const char* op,
               AscendStream& stream);
void transpose(const AscendMat& src, int64_t* perm, AscendMat& dst, AscendStream& stream);
void flip(const AscendMat& src, std::vector<int32_t>& asixs, AscendMat& dst, AscendStream& stream);
void crop(const AscendMat& src, AscendMat& dst, const AscendMat& sizeSrcNpu, int64_t* offset,
          AscendStream& stream);
void transData(const AscendMat& src, AscendMat& dst, const char* from, const char* to,
               AscendStream& stream);
void resize(const AscendMat& src, AscendMat& dst, int32_t* dstSize, int interpolation,
            AscendStream& stream);
} // namespace cann
} // namespace cv

#endif // OPENCV_CANNOPS_CANN_PRIVATE_HPP
