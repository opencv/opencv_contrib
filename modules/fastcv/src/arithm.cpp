/*
 * Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

void matmuls8s32(InputArray _src1, InputArray _src2, OutputArray _dst)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src1.empty() && _src1.type() == CV_8SC1);
    CV_Assert(_src1.cols() <= 131072);
    CV_Assert(_src1.step() % 8 == 0);
    CV_Assert(_src1.cols() == _src2.rows());
    Mat src1 = _src1.getMat();

    CV_Assert(!_src2.empty() && _src2.type() == CV_8SC1);
    CV_Assert(_src2.step() % 8 == 0);
    Mat src2 = _src2.getMat();

    _dst.create(_src1.rows(), _src2.cols(), CV_32SC1);
    // in case of fixed layout array we cannot fix this on our side, can only fail if false
    CV_Assert(_dst.step() % 8 == 0);
    Mat dst = _dst.getMat();

    fcvMatrixMultiplys8s32((const int8_t*)src1.data, src1.cols, src1.rows, src1.step,
                           (const int8_t*)src2.data, src2.cols, src2.step,
                           (int32_t*)dst.data, dst.step);
}

void arithmetic_op(InputArray _src1, InputArray _src2, OutputArray _dst, int op)
{
    CV_Assert(!_src1.empty() && (_src1.depth() == CV_8U || _src1.depth() == CV_16S || _src1.depth() == CV_32F));
    CV_Assert(!_src2.empty() && _src2.type() == _src1.type());
    CV_Assert(_src2.size() == _src1.size());

    Mat src1 = _src1.getMat();
    Mat src2 = _src2.getMat();

    _dst.create(_src1.rows(), _src1.cols(), _src1.type());
    Mat dst = _dst.getMat();

    INITIALIZATION_CHECK;

    fcvConvertPolicy policy = FASTCV_CONVERT_POLICY_SATURATE;

    int nStripes = cv::getNumThreads();

    int func = FCV_OPTYPE(_src1.depth(), op);
    switch(func)
    {
        case FCV_OPTYPE(CV_8U, 0):
            cv::parallel_for_(cv::Range(0, src1.rows), [&](const cv::Range &range){
                          int rangeHeight = range.end - range.start;
                          const uchar* yS1 =  src1.data + static_cast<size_t>(range.start)*src1.step[0];
                          const uchar* yS2 =  src2.data + static_cast<size_t>(range.start)*src2.step[0];
                          uchar* yD = dst.data + static_cast<size_t>(range.start)*dst.step[0];
                          fcvAddu8(yS1, src1.cols, rangeHeight, src1.step[0],
                                     yS2, src2.step[0], policy, yD, dst.step[0]);
                          }, nStripes);
            break;
        case FCV_OPTYPE(CV_16S, 0):
            cv::parallel_for_(cv::Range(0, src1.rows), [&](const cv::Range &range){
                          int rangeHeight = range.end - range.start;
                          const short* yS1 =  (short*)src1.data + static_cast<size_t>(range.start)*(src1.step[0]/sizeof(short));
                          const short* yS2 =  (short*)src2.data + static_cast<size_t>(range.start)*(src2.step[0]/sizeof(short));
                          short* yD = (short*)dst.data + static_cast<size_t>(range.start)*(dst.step[0]/sizeof(short));
                          fcvAdds16_v2(yS1, src1.cols, rangeHeight, src1.step[0],
                                     yS2, src2.step[0], policy, yD, dst.step[0]);
                          }, nStripes);
            break;
        case FCV_OPTYPE(CV_32F, 0):
            cv::parallel_for_(cv::Range(0, src1.rows), [&](const cv::Range &range){
                          int rangeHeight = range.end - range.start;
                          const float* yS1 =  (float*)src1.data + static_cast<size_t>(range.start)*(src1.step[0]/sizeof(float));
                          const float* yS2 =  (float*)src2.data + static_cast<size_t>(range.start)*(src2.step[0]/sizeof(float));
                          float* yD = (float*)dst.data + static_cast<size_t>(range.start)*(dst.step[0]/sizeof(float));
                          fcvAddf32(yS1, src1.cols, rangeHeight, src1.step[0],
                                     yS2, src2.step[0], yD, dst.step[0]);
                          }, nStripes);
            break;
        case FCV_OPTYPE(CV_8U, 1):
            cv::parallel_for_(cv::Range(0, src1.rows), [&](const cv::Range &range){
                          int rangeHeight = range.end - range.start;
                          const uchar* yS1 =  src1.data + static_cast<size_t>(range.start)*src1.step[0];
                          const uchar* yS2 =  src2.data + static_cast<size_t>(range.start)*src2.step[0];
                          uchar* yD = dst.data + static_cast<size_t>(range.start)*dst.step[0];
                          fcvSubtractu8(yS1, src1.cols, rangeHeight, src1.step[0],
                                     yS2, src2.step[0], policy, yD, dst.step[0]);
                          }, nStripes);
            break;
        case FCV_OPTYPE(CV_16S, 1):
            cv::parallel_for_(cv::Range(0, src1.rows), [&](const cv::Range &range){
                          int rangeHeight = range.end - range.start;
                          const short* yS1 =  (short*)src1.data + static_cast<size_t>(range.start)*(src1.step[0]/sizeof(short));
                          const short* yS2 =  (short*)src2.data + static_cast<size_t>(range.start)*(src2.step[0]/sizeof(short));
                          short* yD = (short*)dst.data + static_cast<size_t>(range.start)*(dst.step[0]/sizeof(short));
                          fcvSubtracts16(yS1, src1.cols, rangeHeight, src1.step[0],
                                     yS2, src2.step[0], policy, yD, dst.step[0]);
                          }, nStripes);
            break;
        default:
            CV_Error(cv::Error::StsBadArg, cv::format("op type is not supported"));
            break;
    }
}


void gemm(InputArray _src1, InputArray _src2, OutputArray _dst, float alpha, InputArray _src3, float beta)
{
    CV_Assert(!_src1.empty() && _src1.type() == CV_32FC1);
    CV_Assert(_src1.cols() == _src2.rows());
    Mat src1 = _src1.getMat();

    CV_Assert(!_src2.empty() && _src2.type() == CV_32FC1);
    Mat src2 = _src2.getMat();

    bool isSrc3 = !_src3.empty();

    Mat src3 = _src3.getMat();

    _dst.create(_src1.rows(), _src2.cols(), CV_32FC1);

    Mat dst = _dst.getMat();

    CV_Assert(!FCV_CMP_EQ(alpha,0));

    cv::Mat dst_temp1, dst_temp2;
    float *dstp = NULL;
    bool inplace = false;
    size_t dst_stride;
    fcvStatus status = FASTCV_SUCCESS;

    int n = src1.cols, m = src1.rows, k = src2.cols;

    INITIALIZATION_CHECK;

    if(src1.data == dst.data || src2.data == dst.data || (isSrc3 && (src3.data == dst.data)))
    {
        dst_temp1 = cv::Mat(m, k, CV_32FC1);
        dstp = dst_temp1.ptr<float>();
        inplace = true;
        dst_stride = dst_temp1.step[0];
    }
    else
    {
        dstp = (float32_t*)dst.data;
        dst_stride = dst.step[0];
    }
    float32_t *dstp1 = dstp;
    status = fcvMatrixMultiplyf32_v2((float32_t*)src1.data, n, m, src1.step[0], (float32_t*)src2.data, k,
                                        src2.step[0], dstp, dst_stride);

    bool isAlpha = !(FCV_CMP_EQ(alpha,0) || FCV_CMP_EQ(alpha,1));
    if(isAlpha && status == FASTCV_SUCCESS)
    {
        status = fcvMultiplyScalarf32(dstp, k, m, dst_stride, alpha, dstp1, dst_stride);
    }

    if(isSrc3 && (!FCV_CMP_EQ(beta,0)) && status == FASTCV_SUCCESS)
    {
        cv::Mat dst3 = cv::Mat(m, k, CV_32FC1);
        if(!FCV_CMP_EQ(beta,1))
        {
            status = fcvMultiplyScalarf32((float32_t*)src3.data, k, m, src3.step[0], beta, (float32_t*)dst3.data, dst3.step[0]);
            if(status == FASTCV_SUCCESS)
                fcvAddf32_v2(dstp, k, m, dst_stride, (float32_t*)dst3.data, dst3.step[0], dstp1, dst_stride);
        }
        else
            fcvAddf32_v2(dstp, k, m, dst_stride, (float32_t*)src3.data, src3.step[0], dstp1, dst_stride);
    }

    if(inplace == true)
    {
        dst_temp1(cv::Rect(0, 0, k, m)).copyTo(dst(cv::Rect(0, 0, k, m)));
    }
}

void integrateYUV(InputArray _Y, InputArray _CbCr, OutputArray _IY, OutputArray _ICb, OutputArray _ICr)
{
    CV_Assert(!_Y.empty() && !_CbCr.empty());
    CV_Assert(_Y.type() == _CbCr.type() && _Y.type() == CV_8UC1);
    Mat Y = _Y.getMat();
    Mat CbCr = _CbCr.getMat();
    int Ywidth = Y.cols;
    int Yheight = Y.rows;

    INITIALIZATION_CHECK;

    _IY.create(Yheight + 1, Ywidth + 1, CV_32SC1);
    _ICb.create(Yheight/2 + 1, Ywidth/2 + 1, CV_32SC1);
    _ICr.create(Yheight/2 + 1, Ywidth/2 + 1, CV_32SC1);

    Mat IY_ = _IY.getMat();
    Mat ICb_ = _ICb.getMat();
    Mat ICr_ = _ICr.getMat();

    fcvIntegrateImageYCbCr420PseudoPlanaru8(Y.data, CbCr.data, Ywidth, Yheight, Y.step[0],
                                            CbCr.step[0], (uint32_t*)IY_.data, (uint32_t*)ICb_.data, (uint32_t*)ICr_.data,
                                            IY_.step[0], ICb_.step[0], ICr_.step[0]);
}

} // fastcv::
} // cv::
