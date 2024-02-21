// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv
{
namespace cann
{

static inline void applyMask(const AscendMat& src, AscendMat& dst, const AscendMat& mask,
                             AscendStream& stream)
{
    int mtype = mask.type();
    CV_Assert((mtype == CV_8UC1 || mtype == CV_8SC1) && mask.size() == src.size());
    AscendMat onesMask, castedMask;
    onesMask.create(mask.rows, mask.cols, mask.type());

    OperatorRunner runner;
    runner.setOp("Div")
        .addInput(mask, "x1")
        .addInput(mask, "x2")
        .addOutput(onesMask, "y")
        .run(stream);

    onesMask.convertTo(castedMask, dst.depth(), stream);
    arithm_op(src, castedMask, dst, "Mul", stream);
}

static inline void applyScale(const AscendMat& src, AscendMat& dst, float scale,
                              AscendStream& stream)
{
    OperatorRunner runner;
    arithm_op(src, scale, dst, "Muls", stream);
}

void arithm_op(const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const char* op,
               AscendStream& stream)
{
    if (src2.empty())
        arithm_op(src1, dst, op, stream);
    else
    {
        OperatorRunner runner;
        runner.setOp(op).addInput(src1, "x1").addInput(src2, "x2").addOutput(dst, "y").run(stream);
    }
}

void arithm_op(const AscendMat& src, const Scalar& sc, AscendMat& dst, const char* op,
               AscendStream& stream)
{
    OperatorRunner runner;
    runner.setOp(op)
        .addInput(src, "x1")
        .addInput(sc, src.type(), "x2")
        .addOutput(dst, "y")
        .run(stream);
}

void arithm_op(const Scalar& sc, const AscendMat& src, AscendMat& dst, const char* op,
               AscendStream& stream)
{
    OperatorRunner runner;
    runner.setOp(op)
        .addInput(sc, src.type(), "x1")
        .addInput(src, "x2")
        .addOutput(dst, "y")
        .run(stream);
}

void arithm_op(const AscendMat& src, AscendMat& dst, const char* op, AscendStream& stream)
{
    OperatorRunner runner;
    runner.setOp(op).addInput(src, "x").addOutput(dst, "y").run(stream);
}

void arithm_op(const AscendMat& src, float scalar, AscendMat& dst, const char* op,
               AscendStream& stream)
{
    OperatorRunner runner;
    runner.setOp(op).addInput(src, "x").addAttr(scalar, "value").addOutput(dst, "y").run(stream);
}

// Helper function for template arithm_op. all function called in template arithm_op should be
// done in both AscendMat and Scalar.
static void getInputInfo(const AscendMat& src, int& depth, int& cn, Size& size)
{
    depth = src.depth();
    cn = src.channels();
    size = src.size();
}

static void getInputInfo(const Scalar& src, int& depth, int& cn, Size& size)
{
    CV_UNUSED(src);
    depth = -1;
    cn = -1;
    size = {-1, -1};
}

static void convert(const AscendMat& src, AscendMat& dst, AscendStream& stream)
{
    src.convertTo(dst, CV_32F, stream);
}

static void convert(const Scalar& src, Scalar& dst, AscendStream& stream)
{
    CV_UNUSED(stream);
    dst = src;
}

template <typename T1, typename T2>
static void arithm_op(const T1& src1, const T2& src2, AscendMat& dst, const AscendMat& mask,
                      float scale, int dtype, const char* op, AscendStream& stream)
{
    T1 castedSrc1;
    T2 castedSrc2;
    AscendMat castedRet;

    int sdepth1, sdepth2, scn1, scn2;
    Size ssize1, ssize2;
    getInputInfo(src1, sdepth1, scn1, ssize1);
    getInputInfo(src2, sdepth2, scn2, ssize2);

    int sdepth = sdepth1 == -1 ? sdepth2 : sdepth1;
    int cn = scn1 == -1 ? scn2 : scn1;
    Size size = sdepth1 == -1 ? ssize2 : ssize1;

    if (sdepth1 != -1 && sdepth2 != -1 && !ssize1.empty() && !ssize2.empty())
        CV_Assert(sdepth1 == sdepth2 && scn1 == scn2 && ssize1 == ssize2);

    if (dtype < 0)
        dtype = sdepth;
    const int ddepth = CV_MAT_DEPTH(dtype);
    CV_Assert(sdepth <= CV_16F && ddepth <= CV_16F);

    dst.create(size.height, size.width, CV_MAKE_TYPE(ddepth, cn));

    // In order to achieve high accuracy, convert integers to float for calculation.
    if (scale != 1 && dtype < CV_32F)
    {
        convert(src1, castedSrc1, stream);
        convert(src2, castedSrc2, stream);
        castedRet.create(size.height, size.width, CV_MAKE_TYPE(CV_32F, cn));
    }
    else
    {
        castedSrc1 = src1;
        castedSrc2 = src2;
        castedRet = dst;
    }

    // step1, calculate operator.
    OperatorRunner runner;
    arithm_op(castedSrc1, castedSrc2, castedRet, op, stream);

    // step2, apply mask if need.
    if (!mask.empty())
        applyMask(castedRet, castedRet, mask, stream);

    // step3, apply scale if need.
    if (scale != 1)
        applyScale(castedRet, castedRet, scale, stream);

    // After rounding the result, convert the type to the original type.
    if (castedRet.depth() != dst.depth())
    {
        runner.setOp("Round").addInput(castedRet, "x").addOutput(castedRet, "y").run(stream);
        castedRet.convertTo(dst, stream);
    }
}

static void arithm_op(const InputArray _src1, const InputArray _src2, OutputArray _dst,
                      const InputArray _mask, float scale, int dtype, const char* op,
                      AscendStream& stream)
{
    const bool isScalar1 = (_src1.kind() == _InputArray::MATX);
    const bool isScalar2 = (_src2.kind() == _InputArray::MATX);

    if (isScalar1 && isScalar2)
        CV_Error(Error::StsBadArg, "At list one matrix parameter shoule be passwd.");

    AscendMat src1, src2, dst, mask;
    Mat scalar;

    if (!isScalar1 && !_src1.empty())
        src1.upload(_src1, stream);
    if (!isScalar2 && !_src2.empty())
        src2.upload(_src2, stream);

    if (!_mask.empty())
        mask.upload(_mask, stream);

    Scalar val;
    if (isScalar1)
        scalar = _src1.getMat();
    else if (isScalar2)
        scalar = _src2.getMat();

    if (!scalar.empty())
    {
        CV_Assert(scalar.total() <= 4);
        scalar.convertTo(Mat_<double>(scalar.rows, scalar.cols, &val[0]), CV_64F);
    }

    if (isScalar1)
        arithm_op(val, src2, dst, mask, scale, dtype, op, stream);
    else if (isScalar2)
        arithm_op(src1, val, dst, mask, scale, dtype, op, stream);
    else
        arithm_op(src1, src2, dst, mask, scale, dtype, op, stream);

    dst.download(_dst, stream);
}

// In order to supply more interfaces, differnet function declaration shoule be done.
void add(const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
         int dtype, AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, dtype, "Add", stream);
}

void add(const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
         int dtype, AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, dtype, "Add", stream);
}

void add(const AscendMat& src1, const Scalar& src2, AscendMat& dst, const AscendMat& mask,
         int dtype, AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, dtype, "Add", stream);
}

void add(const Scalar& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
         int dtype, AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, dtype, "Add", stream);
}

void subtract(const InputArray src1, const InputArray src2, OutputArray dst, const InputArray mask,
              int dtype, AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, dtype, "Sub", stream);
}

void subtract(const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
              int dtype, AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, dtype, "Sub", stream);
}

void subtract(const AscendMat& src1, const Scalar& src2, AscendMat& dst, const AscendMat& mask,
              int dtype, AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, dtype, "Sub", stream);
}

void subtract(const Scalar& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
              int dtype, AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, dtype, "Sub", stream);
}

void multiply(const InputArray src1, const InputArray src2, OutputArray dst, float scale, int dtype,
              AscendStream& stream)
{
    arithm_op(src1, src2, dst, noArray(), scale, dtype, "Mul", stream);
}

void multiply(const AscendMat& src1, const AscendMat& src2, AscendMat& dst, float scale, int dtype,
              AscendStream& stream)
{
    arithm_op(src1, src2, dst, AscendMat(), scale, dtype, "Mul", stream);
}

void multiply(const AscendMat& src1, const Scalar& src2, AscendMat& dst, float scale, int dtype,
              AscendStream& stream)
{
    arithm_op(src1, src2, dst, AscendMat(), scale, dtype, "Mul", stream);
}

void multiply(const Scalar& src1, const AscendMat& src2, AscendMat& dst, float scale, int dtype,
              AscendStream& stream)
{
    arithm_op(src1, src2, dst, AscendMat(), scale, dtype, "Mul", stream);
}

void divide(const InputArray src1, const InputArray src2, OutputArray dst, float scale, int dtype,
            AscendStream& stream)
{
    arithm_op(src1, src2, dst, noArray(), scale, dtype, "RealDiv", stream);
}

void divide(const AscendMat& src1, const AscendMat& src2, AscendMat& dst, float scale, int dtype,
            AscendStream& stream)
{
    arithm_op(src1, src2, dst, AscendMat(), scale, dtype, "RealDiv", stream);
}

void divide(const AscendMat& src1, const Scalar& src2, AscendMat& dst, float scale, int dtype,
            AscendStream& stream)
{
    arithm_op(src1, src2, dst, AscendMat(), scale, dtype, "RealDiv", stream);
}

void divide(const Scalar& src1, const AscendMat& src2, AscendMat& dst, float scale, int dtype,
            AscendStream& stream)
{
    arithm_op(src1, src2, dst, AscendMat(), scale, dtype, "RealDiv", stream);
}

void bitwise_and(const InputArray src1, const InputArray src2, OutputArray dst,
                 const InputArray mask, AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseAnd", stream);
}

void bitwise_and(const AscendMat& src1, const AscendMat& src2, AscendMat& dst,
                 const AscendMat& mask, AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseAnd", stream);
}

void bitwise_and(const AscendMat& src1, const Scalar& src2, AscendMat& dst, const AscendMat& mask,
                 AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseAnd", stream);
}

void bitwise_and(const Scalar& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
                 AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseAnd", stream);
}

void bitwise_or(const InputArray src1, const InputArray src2, OutputArray dst,
                const InputArray mask, AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseOr", stream);
}

void bitwise_or(const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
                AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseOr", stream);
}

void bitwise_or(const AscendMat& src1, const Scalar& src2, AscendMat& dst, const AscendMat& mask,
                AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseOr", stream);
}

void bitwise_or(const Scalar& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
                AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseOr", stream);
}

void bitwise_xor(const InputArray src1, const InputArray src2, OutputArray dst,
                 const InputArray mask, AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseXor", stream);
}

void bitwise_xor(const AscendMat& src1, const AscendMat& src2, AscendMat& dst,
                 const AscendMat& mask, AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseXor", stream);
}

void bitwise_xor(const AscendMat& src1, const Scalar& src2, AscendMat& dst, const AscendMat& mask,
                 AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseXor", stream);
}

void bitwise_xor(const Scalar& src1, const AscendMat& src2, AscendMat& dst, const AscendMat& mask,
                 AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseXor", stream);
}

void bitwise_not(const InputArray src, OutputArray dst, const InputArray mask, AscendStream& stream)
{
    arithm_op(src, noArray(), dst, mask, 1, -1, "Invert", stream);
}

void bitwise_not(const AscendMat& src, AscendMat& dst, const AscendMat& mask, AscendStream& stream)
{
    arithm_op(src, AscendMat(), dst, mask, 1, -1, "Invert", stream);
}

void addWeighted(const AscendMat& src1, double alpha, const AscendMat& src2, double beta,
                 double gamma, AscendMat& dst, int dtype, AscendStream& stream)
{
    if (dtype < 0)
        dtype = src1.depth();

    CV_Assert(src2.depth() == src1.depth() && src2.size() == src1.size() &&
              src1.channels() == src2.channels());

    int type = CV_MAKE_TYPE(dtype, src1.channels());
    dst.create(src1.rows, src1.cols, type);

    // TODO: Consider overflow, should extend type or not?
    AscendMat src1Weighted(src1.size(), type), src2Weighted(src1.size(), type),
        srcWeightedSumRet(src1.size(), type);

    arithm_op(src1, (float)alpha, src1Weighted, "Muls", stream);
    arithm_op(src2, (float)beta, src2Weighted, "Muls", stream);
    arithm_op(src1Weighted, src2Weighted, srcWeightedSumRet, "Add", stream);
    arithm_op(srcWeightedSumRet, (float)gamma, dst, "Adds", stream);
}

void addWeighted(const InputArray _src1, double alpha, const InputArray _src2, double beta,
                 double gamma, OutputArray _dst, int dtype, AscendStream& stream)
{
    AscendMat src1, src2, dst;
    src1.upload(_src1, stream);
    src2.upload(_src2, stream);
    addWeighted(src1, alpha, src2, beta, gamma, dst, dtype, stream);
    dst.download(_dst, stream);
}

double threshold(const AscendMat& src, AscendMat& dst, double thresh, double maxval, int type,
                 AscendStream& stream)
{
    // ThresholdTypes is defined in opencv2/imgproc, This type is the only Symbol we need.
    // Add imgproc to dependence is too heavy, use magic number instead.
    CV_Assert(type <= 4 /*THRESH_TOZERO_INV*/);

    AscendMat threshMat(src.size(), src.type());

    dst.create(src.rows, src.cols, src.type());

    if (src.depth() == CV_8U || src.depth() == CV_8S || src.depth() == CV_16S ||
        src.depth() == CV_32S || src.depth() == CV_32F || src.depth() == CV_16F)
    {
        ThresholdOpencvTilingData tiling;
        tiling.maxVal = maxval;
        tiling.thresh = thresh;
        // AscendMat memory will be align to 32B, it's safe to set totalLengh a little bigger.
        size_t totalBytes = src.rows * src.cols * src.channels();
        tiling.totalLength = ALIGN_UP(totalBytes, 32);
        tiling.threshType = type;
        tiling.dtype = src.depth();

        kernel_launch(aclrtlaunch_threshold_opencv, stream, tiling, src.data.get(), dst.data.get());
    }
    else
        CV_Error(Error::StsUnsupportedFormat, "");

    return thresh;
}

double threshold(const InputArray _src, OutputArray _dst, double thresh, double maxval, int type,
                 AscendStream& stream)
{
    AscendMat src, dst;
    src.upload(_src, stream);
    dst.create(src.rows, src.cols, src.type());
    double ret = threshold(src, dst, thresh, maxval, type, stream);
    dst.download(_dst, stream);
    return ret;
}

} // namespace cann
} // namespace cv
