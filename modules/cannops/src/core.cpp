// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv
{
namespace cann
{
// Transform data type from one to another. eg. from NCHW to NHWC.
void transData(const AscendMat& src, AscendMat& dst, const char* from, const char* to,
               AscendStream& stream)
{
    OperatorRunner runner;
    runner.setOp("TransData")
        .addInput(src, "src")
        .addOutput(dst, "dst")
        .addAttr(from, "src_format")
        .addAttr(to, "dst_format")
        .run(stream);
}

void merge(const AscendMat* src, size_t n, AscendMat& dst, AscendStream& stream)
{
    if (src == nullptr || n < 2)
        return;

    int depth = src->depth();
    int rows = src->rows;
    int cols = src->cols;

    // All matrix must have same size and type
    for (size_t i = 1; i < n; i++)
    {
        CV_Assert(src[i].depth() == depth && src[i].channels() == 1);
        CV_Assert(src[i].rows == rows && src[i].cols == cols);
    }

    int cns = 0;
    for (size_t i = 0; i < n; i++)
        cns += src[i].channels();
    dst.create(src->rows, src->cols, CV_MAKE_TYPE(src->depth(), cns));

    OperatorRunner runner;
    runner.setOp("ConcatD");

    for (size_t i = 0; i < n; i++)
    {
        runner.addInput(src[i], ("x" + std::to_string(i)).c_str());
    }

    runner.addOutput(dst, "output_data").addAttr(3, "concat_dim").run(stream);
}

void merge(const std::vector<AscendMat>& src, AscendMat& dst, AscendStream& stream)
{
    merge(&src[0], src.size(), dst, stream);
}

void merge(const AscendMat* src, size_t n, OutputArray& _dst, AscendStream& stream)
{
    AscendMat dst;
    merge(src, n, dst, stream);
    dst.download(_dst, stream);
}
void merge(const std::vector<AscendMat>& src, OutputArray& dst, AscendStream& stream)
{
    merge(&src[0], src.size(), dst, stream);
}

void split(const AscendMat& src, AscendMat* dst, AscendStream& stream)
{
    if (src.empty() || dst == nullptr)
        return;

    int cn = src.channels();

    OperatorRunner runner;
    runner.setOp("SplitD").addInput(src, "x");
    for (int i = 0; i < cn; i++)
    {
        dst[i].create(src.rows, src.cols, CV_MAKE_TYPE(src.depth(), 1));
        runner.addOutput(dst[i], ("y" + std::to_string(i)).c_str());
    }
    runner.addAttr(3, "split_dim").addAttr(cn, "num_split").run(stream);
}

void split(const AscendMat& src, std::vector<AscendMat>& dst, AscendStream& stream)
{
    dst.resize(src.channels());
    split(src, &dst[0], stream);
}

void split(const InputArray _src, AscendMat* dst, AscendStream& stream)
{
    AscendMat src;
    src.upload(_src, stream);
    split(src, dst, stream);
}
void split(const InputArray _src, std::vector<AscendMat>& dst, AscendStream& stream)
{
    AscendMat src;
    src.upload(_src, stream);
    dst.resize(src.channels());
    split(_src, &dst[0], stream);
}

void transpose(const AscendMat& src, int64_t* perm, AscendMat& dst, AscendStream& stream)
{
    OperatorRunner runner;
    runner.setOp("TransposeD")
        .addInput(src, "x")
        .addOutput(dst, "y")
        .addAttr(perm, 4, "perm")
        .run(stream);
}

void transpose(const AscendMat& src, AscendMat& dst, AscendStream& stream)
{
    int64_t perm[] = {0, 2, 1, 3};
    dst.create(src.cols, src.rows, src.type());
    transpose(src, perm, dst, stream);
}

void transpose(InputArray _src, OutputArray _dst, AscendStream& stream)
{
    AscendMat src, dst;
    src.upload(_src, stream);
    transpose(src, dst, stream);
    dst.download(_dst, stream);
}

void flip(const AscendMat& src, std::vector<int32_t>& asixs, AscendMat& dst, AscendStream& stream)
{
    int64_t dim = asixs.size();
    OperatorRunner runner;
    runner.setOp("ReverseV2")
        .addInput(src, "x")
        .addInput<int32_t>(&asixs.at(0), &dim, 1, ACL_INT32, "axis")
        .addOutput(dst, "y")
        .run(stream);
}

void flip(const AscendMat& src, AscendMat& dst, int flipCode, AscendStream& stream)
{
    std::vector<int32_t> asix;
    if (flipCode == 0)
        asix.push_back(1);
    else if (flipCode > 0)
        asix.push_back(2);
    else
    {
        asix.push_back(1);
        asix.push_back(2);
    }
    dst.create(src.rows, src.cols, src.type());
    flip(src, asix, dst, stream);
}

void flip(const InputArray _src, OutputArray _dst, int flipCode, AscendStream& stream)
{
    AscendMat src, dst;
    src.upload(_src, stream);
    flip(src, dst, flipCode, stream);
    dst.download(_dst, stream);
}

void rotate(const AscendMat& src, AscendMat& dst, int rotateMode, AscendStream& stream)
{
    AscendMat tempMat;
    switch (rotateMode)
    {
        case ROTATE_90_CLOCKWISE:
        {
            dst.create(src.cols, src.rows, src.type());
            transpose(src, tempMat, stream);
            flip(tempMat, dst, 1, stream);
            break;
        }
        case ROTATE_180:
        {
            dst.create(src.rows, src.cols, src.type());
            flip(src, dst, -1, stream);
            break;
        }
        case ROTATE_90_COUNTERCLOCKWISE:
        {
            dst.create(src.cols, src.rows, src.type());
            transpose(src, tempMat, stream);
            flip(tempMat, dst, 0, stream);
            break;
        }
        default:
            break;
    }
}

void rotate(InputArray _src, OutputArray _dst, int rotateMode, AscendStream& stream)
{
    AscendMat src, dst;
    src.upload(_src, stream);
    rotate(src, dst, rotateMode, stream);
    dst.download(_dst, stream);
}

void crop(const AscendMat& src, AscendMat& dst, const AscendMat& sizeSrcNpu, int64_t* offset,
          AscendStream& stream)
{
    OperatorRunner runner;
    runner.setOp("Crop")
        .addInput(src, "x")
        .addInput(sizeSrcNpu, "size")
        .addAttr(1, "axis")
        .addAttr(offset, 3, "offsets")
        .addOutput(dst, "y")
        .run(stream);
}

AscendMat crop(const AscendMat& src, const Rect& rect, AscendStream& stream)
{
    AscendMat dst, sizeSrcNpu;
    // left-up conner
    int x = rect.x, y = rect.y, width = rect.width, height = rect.height;
    int64_t offset[] = {y, x, 0};

    CV_Assert(x + width <= src.cols && y + height <= src.rows);
    int size1[] = {1, src.channels(), height, width};
    dst.create(height, width, src.type());

    Mat sizeSrc(height, width, src.type(), size1);
    sizeSrcNpu.upload(sizeSrc);
    crop(src, dst, sizeSrcNpu, offset, stream);

    return dst;
}
AscendMat crop(InputArray _src, const Rect& rect, AscendStream& stream)
{
    AscendMat src;
    src.upload(_src, stream);
    return crop(src, rect, stream);
}

void resize(const AscendMat& src, AscendMat& dst, int32_t* dstSize, int interpolation,
            AscendStream& stream)
{
    OperatorRunner runner;
    int64_t dims[] = {2};
    char const* mode = "";
    switch (interpolation)
    {
        case INTER_CUBIC:
            mode = "ResizeBicubic";
            break;
        case INTER_AREA:
            mode = "ResizeArea";
            break;
        default:
            break;
    }

    runner.setOp(mode)
        .addInput(src, "images")
        .addInput<int32_t>(dstSize, dims, 1, ACL_INT32, "size")
        .addAttr(true, "half_pixel_centers")
        .addOutput(dst, "y")
        .run(stream);
}

void resize(const AscendMat& src, AscendMat& dst, Size dsize, double inv_scale_x,
            double inv_scale_y, int interpolation, AscendStream& stream)
{
    Size ssize = src.size();
    CV_Assert(!ssize.empty());
    float_t scaleX = (float_t)inv_scale_x;
    float_t scaleY = (float_t)inv_scale_y;
    CV_Assert(interpolation == INTER_CUBIC || interpolation == INTER_AREA);

    if (dsize.empty())
    {
        CV_Assert(scaleX > 0);
        CV_Assert(scaleY > 0);
        dsize = Size(saturate_cast<int>(ssize.width * inv_scale_x),
                     saturate_cast<int>(ssize.height * inv_scale_y));
        CV_Assert(!dsize.empty());
    }
    else
    {
        scaleX = (float_t)dsize.width / ssize.width;
        scaleY = (float_t)dsize.height / ssize.height;
        CV_Assert(scaleX > 0);
        CV_Assert(scaleY > 0);
    }

    int32_t dstSize[] = {dsize.width, dsize.height};
    dst.create(dstSize[0], dstSize[1], src.type());
    resize(src, dst, dstSize, interpolation, stream);
}

void resize(InputArray _src, OutputArray _dst, Size dsize, double inv_scale_x, double inv_scale_y,
            int interpolation, AscendStream& stream)
{
    AscendMat src, dst;
    src.upload(_src, stream);
    resize(src, dst, dsize, inv_scale_x, inv_scale_y, interpolation, stream);
    dst.download(_dst, stream);
}

} // namespace cann
} // namespace cv
