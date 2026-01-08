/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

static void trackOpticalFlowLKInternal(InputArray _src, InputArray _dst,
                                       InputArrayOfArrays _srcPyr, InputArrayOfArrays _dstPyr,
                                       InputArrayOfArrays _srcDxPyr, InputArrayOfArrays _srcDyPyr,
                                       InputArray _ptsIn, OutputArray _ptsOut, InputArray _ptsEst,
                                       OutputArray _statusVec, cv::Size winSize,
                                       cv::TermCriteria termCriteria)
{
    INITIALIZATION_CHECK;

    CV_Assert(winSize.width % 2 == 1 && winSize.height % 2 == 1);

    CV_Assert(!_src.empty() && _src.type() == CV_8UC1);
    CV_Assert(!_dst.empty() && _dst.type() == CV_8UC1);
    CV_Assert(_src.size() == _dst.size());
    CV_Assert(_src.step() % 8 == 0);
    CV_Assert(_dst.step() == _src.step());

    cv::Mat src = _src.getMat(), dst = _dst.getMat();

    CV_Assert(_srcPyr.kind() == _InputArray::KindFlag::STD_ARRAY_MAT ||
              _srcPyr.kind() == _InputArray::KindFlag::STD_VECTOR_MAT ||
              _srcPyr.kind() == _InputArray::KindFlag::STD_VECTOR_UMAT);
    CV_Assert(_dstPyr.kind() == _InputArray::KindFlag::STD_ARRAY_MAT ||
              _dstPyr.kind() == _InputArray::KindFlag::STD_VECTOR_MAT ||
              _dstPyr.kind() == _InputArray::KindFlag::STD_VECTOR_UMAT);
    CV_Assert(_srcPyr.size() == _dstPyr.size());

    int nLevels = _srcPyr.size().area();

    std::vector<cv::Mat> srcPyr, dstPyr;
    _srcPyr.getMatVector(srcPyr);
    _dstPyr.getMatVector(dstPyr);

    cv::Size imSz = src.size();
    for (int i = 0; i < nLevels; i++)
    {
        const cv::Mat& s = srcPyr[i];
        const cv::Mat& d = dstPyr[i];

        CV_Assert(!s.empty() && s.type() == CV_8UC1);
        CV_Assert(!d.empty() && d.type() == CV_8UC1);
        CV_Assert(s.size() == imSz);
        CV_Assert(d.size() == imSz);

        imSz.width /= 2; imSz.height /= 2;
    }

    bool useDxDy = !_srcDxPyr.empty() && !_srcDyPyr.empty();
    int version = useDxDy ? 1 : 3;

    std::vector<cv::Mat> srcDxPyr, srcDyPyr;
    if (version == 1)
    {
        CV_Assert(_srcDxPyr.kind() == _InputArray::KindFlag::STD_ARRAY_MAT ||
                  _srcDxPyr.kind() == _InputArray::KindFlag::STD_VECTOR_MAT ||
                  _srcDxPyr.kind() == _InputArray::KindFlag::STD_VECTOR_UMAT);
        CV_Assert(_srcDyPyr.kind() == _InputArray::KindFlag::STD_ARRAY_MAT ||
                  _srcDyPyr.kind() == _InputArray::KindFlag::STD_VECTOR_MAT ||
                  _srcDyPyr.kind() == _InputArray::KindFlag::STD_VECTOR_UMAT);

        CV_Assert(_srcDxPyr.size() == _srcDyPyr.size());
        _srcDxPyr.getMatVector(srcDxPyr);
        _srcDyPyr.getMatVector(srcDyPyr);

        imSz = src.size();
        for (int i = 0; i < nLevels; i++)
        {
            const cv::Mat& dx = srcDxPyr[i];
            const cv::Mat& dy = srcDyPyr[i];

            CV_Assert(!dx.empty() && dx.type() == CV_8SC1);
            CV_Assert(!dy.empty() && dy.type() == CV_8SC1);
            CV_Assert(dx.size() == imSz);
            CV_Assert(dy.size() == imSz);

            imSz.width /= 2; imSz.height /= 2;
        }
    }

    std::vector<fcvPyramidLevel> lpyrSrc1, lpyrDst1, lpyrDxSrc, lpyrDySrc;
    std::vector<fcvPyramidLevel_v2> lpyrSrc2, lpyrDst2;
    for (int i = 0; i < nLevels; i++)
    {
        fcvPyramidLevel lsrc1, ldst1;
        fcvPyramidLevel_v2 lsrc2, ldst2;
        lsrc1.width  = srcPyr[i].cols;
        lsrc1.height = srcPyr[i].rows;
        lsrc1.ptr    = srcPyr[i].data;

        lsrc2.width  = srcPyr[i].cols;
        lsrc2.height = srcPyr[i].rows;
        lsrc2.stride = srcPyr[i].step;
        lsrc2.ptr    = srcPyr[i].data;

        ldst1.width  = dstPyr[i].cols;
        ldst1.height = dstPyr[i].rows;
        ldst1.ptr    = dstPyr[i].data;
        ldst2.width  = dstPyr[i].cols;
        ldst2.height = dstPyr[i].rows;
        ldst2.stride = dstPyr[i].step;
        ldst2.ptr    = dstPyr[i].data;
        lpyrSrc1.push_back(lsrc1); lpyrDst1.push_back(ldst1);
        lpyrSrc2.push_back(lsrc2); lpyrDst2.push_back(ldst2);

        if (version == 1)
        {
            fcvPyramidLevel ldx, ldy;
            CV_Assert(srcDxPyr[i].isContinuous());
            ldx.width  = srcDxPyr[i].cols;
            ldx.height = srcDxPyr[i].rows;
            ldx.ptr    = srcDxPyr[i].data;
            CV_Assert(srcDyPyr[i].isContinuous());
            ldy.width  = srcDyPyr[i].cols;
            ldy.height = srcDyPyr[i].rows;
            ldy.ptr    = srcDyPyr[i].data;
            lpyrDxSrc.push_back(ldx); lpyrDySrc.push_back(ldy);
        }
    }

    CV_Assert(!_ptsIn.empty() && (_ptsIn.type() == CV_32FC1 || _ptsIn.type() == CV_32FC2));
    CV_Assert(_ptsIn.isContinuous());
    CV_Assert(_ptsIn.total() * _ptsIn.channels() % 2 == 0);

    cv::Mat ptsIn = _ptsIn.getMat();
    int nPts = ptsIn.total() * ptsIn.channels() / 2;

    bool useInitialEstimate;
    cv::Mat ptsEst;
    const float32_t* ptsEstData;
    if (!_ptsEst.empty())
    {
        CV_Assert(_ptsEst.type() == CV_32FC1 || _ptsEst.type() == CV_32FC2);
        CV_Assert(_ptsEst.isContinuous());
        int estElems = _ptsEst.total() * _ptsEst.channels();
        CV_Assert(estElems % 2 == 0);
        CV_Assert(estElems / 2 == nPts);

        ptsEst = _ptsEst.getMat();
        ptsEstData = (const float32_t*)ptsEst.data;
        useInitialEstimate = true;
    }
    else
    {
        useInitialEstimate = false;
        ptsEstData = (const float32_t*)ptsIn.data;
    }

    CV_Assert(_ptsOut.needed());
    _ptsOut.create(1, nPts, CV_32FC2);
    cv::Mat ptsOut = _ptsOut.getMat();

    cv::Mat statusVec;
    if (!_statusVec.empty())
    {
        _statusVec.create(1, nPts, CV_32SC1);
        statusVec = _statusVec.getMat();
    }
    else
    {
        statusVec = cv::Mat(1, nPts, CV_32SC1);
    }

    fcvTerminationCriteria termCrit;
    if (termCriteria.type & cv::TermCriteria::COUNT)
    {
        if (termCriteria.type & cv::TermCriteria::EPS)
        {
            termCrit = FASTCV_TERM_CRITERIA_BOTH;
        }
        else
        {
            termCrit = FASTCV_TERM_CRITERIA_ITERATIONS;
        }
    }
    else
    {
        if (termCriteria.type & cv::TermCriteria::EPS)
        {
            termCrit = FASTCV_TERM_CRITERIA_EPSILON;
        }
        else
        {
            CV_Error(cv::Error::StsBadArg, "Incorrect termination criteria");
        }
    }
    int maxIterations = termCriteria.maxCount;
    double maxEpsilon = termCriteria.epsilon;

    fcvStatus status = FASTCV_SUCCESS;

    if (version == 3)
    {
        status = fcvTrackLKOpticalFlowu8_v3(src.data, dst.data, src.cols, src.rows, src.step,
                                            lpyrSrc2.data(), lpyrDst2.data(),
                                            (const float32_t*)ptsIn.data,
                                            ptsEstData,
                                            (float32_t*)ptsOut.data,
                                            (int32_t*)statusVec.data,
                                            nPts,
                                            winSize.width, winSize.height,
                                            nLevels,
                                            termCrit, maxIterations, maxEpsilon,
                                            useInitialEstimate);
    }
    else // if (version == 1)
    {
        CV_Assert(src.isContinuous() && dst.isContinuous());
        // Obsolete parameters, set to 0
        float maxResidue = 0, minDisplacement = 0, minEigenvalue = 0;
        int lightingNormalized = 0;
        fcvTrackLKOpticalFlowu8(src.data, dst.data, src.cols, src.rows,
                                lpyrSrc1.data(), lpyrDst1.data(),
                                lpyrDxSrc.data(), lpyrDySrc.data(),
                                (const float32_t*)ptsIn.data,
                                (float32_t*)ptsOut.data,
                                (int32_t*)statusVec.data,
                                nPts,
                                winSize.width, winSize.height,
                                maxIterations,
                                nLevels,
                                maxResidue, minDisplacement, minEigenvalue, lightingNormalized);
    }

    if (status != FASTCV_SUCCESS)
    {
        std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
        CV_Error( cv::Error::StsInternal, "FastCV error: " + s);
    }
}


void trackOpticalFlowLK(InputArray _src, InputArray _dst,
                        InputArrayOfArrays _srcPyr, InputArrayOfArrays _dstPyr,
                        InputArray _ptsIn, OutputArray _ptsOut, InputArray _ptsEst,
                        OutputArray _statusVec, cv::Size winSize,
                        cv::TermCriteria termCriteria)
{
    trackOpticalFlowLKInternal(_src, _dst, _srcPyr, _dstPyr, noArray(), noArray(),
                               _ptsIn, _ptsOut, _ptsEst,
                               _statusVec, winSize,
                               termCriteria);
}

void trackOpticalFlowLK(InputArray _src, InputArray _dst,
                        InputArrayOfArrays _srcPyr, InputArrayOfArrays _dstPyr,
                        InputArrayOfArrays _srcDxPyr, InputArrayOfArrays _srcDyPyr,
                        InputArray _ptsIn, OutputArray _ptsOut,
                        OutputArray _statusVec, cv::Size winSize, int maxIterations)
{
    trackOpticalFlowLKInternal(_src, _dst, _srcPyr, _dstPyr,
                               _srcDxPyr, _srcDyPyr,
                               _ptsIn, _ptsOut, cv::noArray(),
                               _statusVec, winSize,
                               {cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                                maxIterations, /* maxEpsilon */ 0.03f * 0.03f});
}

} // fastcv::
} // cv::
