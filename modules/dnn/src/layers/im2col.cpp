#include "../precomp.hpp"
#include <opencv2/core/ocl.hpp>
#include "im2col.hpp"
#include "opencl_kernels_dnn.hpp"

namespace cv
{
namespace dnn
{

void im2col_ocl(UMat &img,
                int channels, int height, int width,
                int kernel_h, int kernel_w,
                int pad_h, int pad_w,
                int stride_h, int stride_w,
                UMat &col)
{
    int h_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int w_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    CV_Assert(img.isContinuous() && col.isContinuous());
    CV_Assert(img.total() == (size_t)channels * height * width);
    CV_Assert(col.total() == (size_t)channels * kernel_h * kernel_w * h_out * w_out);

    ocl::Kernel im2col_ker("im2col", ocl::dnn::im2col_oclsrc);
    CV_Assert(!im2col_ker.empty());

    im2col_ker.args(ocl::KernelArg::PtrReadOnly(img), (int)img.offset,
             channels, height, width,
             kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
             h_out, w_out,
             ocl::KernelArg::PtrWriteOnly(col), (int)col.offset
        );

    size_t localSize = ocl::Device::getDefault().maxWorkGroupSize();
    size_t globalSize = (size_t)channels * h_out * w_out;

    CV_Assert(im2col_ker.run(1, &globalSize, &localSize, true));
}

}
}
