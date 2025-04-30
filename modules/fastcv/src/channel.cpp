/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

void merge(InputArrayOfArrays _mv, OutputArray _dst)
{
    CV_Assert(!_mv.empty());
    std::vector<cv::Mat> mv;
    _mv.getMatVector(mv);
    int count = mv.size();

    CV_Assert(!mv.empty());

    CV_Assert(count == 2 || count == 3 || count == 4);
    CV_Assert(!mv[0].empty());
    CV_Assert(mv[0].dims <= 2);

    for(int i = 0; i < count; i++ )
    {
        CV_Assert(mv[i].size == mv[0].size && mv[i].step[0] == mv[0].step[0] && mv[i].type() == CV_8UC1);
    }

     _dst.create(mv[0].dims, mv[0].size, CV_MAKE_TYPE(CV_8U,count));
    Mat dst = _dst.getMat();

    INITIALIZATION_CHECK;

    int nStripes = cv::getNumThreads();

    switch(count)
    {
        case 2:
        cv::parallel_for_(cv::Range(0, mv[0].rows), [&](const cv::Range &range){
                          int height_ = range.end - range.start;
                          const uchar* yS1 =  mv[0].data + static_cast<size_t>(range.start) * mv[0].step[0];
                          const uchar* yS2 =  mv[1].data + static_cast<size_t>(range.start) * mv[1].step[0];
                          uchar* yD = dst.data + static_cast<size_t>(range.start) * dst.step[0];
                          fcvChannelCombine2Planesu8(yS1, mv[0].cols, height_, mv[0].step[0], yS2, mv[1].step[0], yD, dst.step[0]);
                          }, nStripes);

        break;

        case 3:
        cv::parallel_for_(cv::Range(0, mv[0].rows), [&](const cv::Range &range){
                          int height_ = range.end - range.start;
                          const uchar* yS1 =  mv[0].data + static_cast<size_t>(range.start) * mv[0].step[0];
                          const uchar* yS2 =  mv[1].data + static_cast<size_t>(range.start) * mv[1].step[0];
                          const uchar* yS3 =  mv[2].data + static_cast<size_t>(range.start) * mv[2].step[0];
                          uchar* yD = dst.data + static_cast<size_t>(range.start) * dst.step[0];
                          fcvChannelCombine3Planesu8(yS1, mv[0].cols, height_, mv[0].step[0], yS2, mv[1].step[0], yS3, mv[2].step[0], yD, dst.step[0]);
                          }, nStripes);

        break;

        case 4:
        cv::parallel_for_(cv::Range(0, mv[0].rows), [&](const cv::Range &range){
                          int height_ = range.end - range.start;
                          const uchar* yS1 =  mv[0].data + static_cast<size_t>(range.start) * mv[0].step[0];
                          const uchar* yS2 =  mv[1].data + static_cast<size_t>(range.start) * mv[1].step[0];
                          const uchar* yS3 =  mv[2].data + static_cast<size_t>(range.start) * mv[2].step[0];
                          const uchar* yS4 =  mv[3].data + static_cast<size_t>(range.start) * mv[3].step[0];
                          uchar* yD = dst.data + static_cast<size_t>(range.start) * dst.step[0];
                          fcvChannelCombine4Planesu8(yS1, mv[0].cols, height_, mv[0].step[0], yS2, mv[1].step[0], yS3, mv[2].step[0], yS4, mv[3].step[0], yD, dst.step[0]);
                          }, nStripes);

        break;

        default:
        CV_Error(cv::Error::StsBadArg, cv::format("count is not supported"));
        break;
    }
}

void split(InputArray _src, OutputArrayOfArrays _mv)
{
    CV_Assert(!_src.empty());
    Mat src = _src.getMat();

    int depth = src.depth(), cn = src.channels();

    CV_Assert(depth == CV_8U && (cn == 2 || cn == 3 || cn == 4));
    CV_Assert(src.dims <= 2);
    _mv.create(cn, 1, depth);
    for( int k = 0; k < cn; k++ )
    {
        _mv.create(src.dims, src.size, depth, k);
    }

    std::vector<cv::Mat> mv(cn);
    _mv.getMatVector(mv);

    INITIALIZATION_CHECK;

    int nStripes = cv::getNumThreads();

    if(src.rows * src.cols < 640 * 480)
        if(cn == 3 || cn == 4)
            nStripes = 1;

    if(cn == 2)
    {
        cv::parallel_for_(cv::Range(0, src.rows), [&](const cv::Range &range){
                      int height_ = range.end - range.start;
                      const uchar* yS =  src.data + static_cast<size_t>(range.start) * src.step[0];
                      uchar* y1D = mv[0].data + static_cast<size_t>(range.start) * mv[0].step[0];
                      uchar* y2D = mv[1].data + static_cast<size_t>(range.start) * mv[1].step[0];
                      fcvDeinterleaveu8(yS, src.cols, height_, src.step[0], y1D, mv[0].step[0], y2D, mv[1].step[0]);
                      }, nStripes);
    }
    else if(cn == 3)
    {
        for(int i=0; i<cn; i++)
        {
            cv::parallel_for_(cv::Range(0, src.rows), [&](const cv::Range &range){
                      int height_ = range.end - range.start;
                      const uchar* yS =  src.data + static_cast<size_t>(range.start) * src.step[0];
                      uchar* yD = mv[i].data + static_cast<size_t>(range.start) * mv[i].step[0];
                      fcvChannelExtractu8(yS, src.cols, height_, src.step[0], NULL, 0, NULL, 0, (fcvChannelType)i, (fcvImageFormat)FASTCV_RGB, yD, mv[i].step[0]);
                      }, nStripes);
        }
    }
    else if(cn == 4)
    {
        for(int i=0; i<cn; i++)
        {
            cv::parallel_for_(cv::Range(0, src.rows), [&](const cv::Range &range){
                      int height_ = range.end - range.start;
                      const uchar* yS =  src.data + static_cast<size_t>(range.start) * src.step[0];
                      uchar* yD = mv[i].data + static_cast<size_t>(range.start) * mv[i].step[0];
                      fcvChannelExtractu8(yS, src.cols, height_, src.step[0], NULL, 0, NULL, 0, (fcvChannelType)i, (fcvImageFormat)FASTCV_RGBX, yD, mv[i].step[0]);
                      }, nStripes);
        }
    }
}

} // fastcv::
} // cv::
