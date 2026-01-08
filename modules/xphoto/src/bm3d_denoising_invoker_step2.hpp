/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#ifndef __OPENCV_BM3D_DENOISING_INVOKER_STEP2_HPP__
#define __OPENCV_BM3D_DENOISING_INVOKER_STEP2_HPP__

#include "bm3d_denoising_invoker_commons.hpp"
#include "bm3d_denoising_transforms.hpp"
#include "kaiser_window.hpp"

namespace cv
{
namespace xphoto
{

template <typename T, typename D, typename WT, typename TT, typename TC>
struct Bm3dDenoisingInvokerStep2 : public ParallelLoopBody
{
public:
    Bm3dDenoisingInvokerStep2(
        const Mat& src,
        const Mat& basic,
        Mat& dst,
        const int &templateWindowSize,
        const int &searchWindowSize,
        const float &h,
        const int &hBM,
        const int &groupSize,
        const int &slidingStep,
        const float &beta);

    virtual ~Bm3dDenoisingInvokerStep2();
    void operator() (const Range& range) const CV_OVERRIDE;

private:
    // Unimplemented operator in order to satisfy compiler warning.
    void operator= (const Bm3dDenoisingInvokerStep2&);

    void calcDistSumsForFirstElementInRow(
        int i,
        Array2d<int>& distSums,
        Array3d<int>& colDistSums,
        Array3d<int>& lastColDistSums,
        BlockMatch<TT, int, TT> *bm,
        int &elementSize) const;

    void calcDistSumsForAllElementsInFirstRow(
        int i,
        int j,
        int firstColNum,
        Array2d<int>& distSums,
        Array3d<int>& colDistSums,
        Array3d<int>& lastColDistSums,
        BlockMatch<TT, int, TT> *bm,
        int &elementSize) const;

    // Image containers
    const Mat& src_;
    const Mat& basic_;
    Mat& dst_;
    Mat srcExtended_;
    Mat basicExtended_;

    // Border size of the extended src and basic images
    int borderSize_;

    // Template and window size
    int templateWindowSize_;
    int searchWindowSize_;

    // Half template and window size
    int halfTemplateWindowSize_;
    int halfSearchWindowSize_;

    // Squared template and window size
    int templateWindowSizeSq_;
    int searchWindowSizeSq_;

    // Block matching threshold
    int hBM_;

    // Maximum size of 3D group
    int groupSize_;

    // Sliding step
    const int slidingStep_;

    // Threshold map
    TT *thrMap_;

    // Kaiser window
    float *kaiser_;
};

template <typename T, typename D, typename WT, typename TT, typename TC>
Bm3dDenoisingInvokerStep2<T, D, WT, TT, TC>::Bm3dDenoisingInvokerStep2(
    const Mat& src,
    const Mat& basic,
    Mat& dst,
    const int &templateWindowSize,
    const int &searchWindowSize,
    const float &h,
    const int &hBM,
    const int &groupSize,
    const int &slidingStep,
    const float &beta) :
    src_(src), basic_(basic), dst_(dst), groupSize_(groupSize), slidingStep_(slidingStep), thrMap_(NULL), kaiser_(NULL)
{
    groupSize_ = getLargestPowerOf2SmallerThan(groupSize);
    CV_Assert(groupSize > 0);

    halfTemplateWindowSize_ = templateWindowSize >> 1;
    halfSearchWindowSize_ = searchWindowSize >> 1;
    templateWindowSize_ = templateWindowSize;
    searchWindowSize_ = searchWindowSize;
    templateWindowSizeSq_ = templateWindowSize_ * templateWindowSize_;
    searchWindowSizeSq_ = searchWindowSize_ * searchWindowSize_;

    // Extend image to avoid border problem
    borderSize_ = halfSearchWindowSize_ + halfTemplateWindowSize_;
    copyMakeBorder(src_, srcExtended_, borderSize_, borderSize_, borderSize_, borderSize_, BORDER_DEFAULT);
    copyMakeBorder(basic_, basicExtended_, borderSize_, borderSize_, borderSize_, borderSize_, BORDER_DEFAULT);

    // Calculate block matching threshold
    hBM_ = D::template calcBlockMatchingThreshold<int>(hBM, templateWindowSizeSq_);

    // Select transforms depending on the template size
    TC::RegisterTransforms2D(templateWindowSize_);

    // Precompute threshold map
    TC::calcThresholdMap3D(thrMap_, h, templateWindowSize_, groupSize_);

    // Generate kaiser window
    calcKaiserWindow2D(kaiser_, templateWindowSize_, beta);
}

template<typename T, typename D, typename WT, typename TT, typename TC>
inline Bm3dDenoisingInvokerStep2<T, D, WT, TT, TC>::~Bm3dDenoisingInvokerStep2()
{
    delete[] thrMap_;
    delete[] kaiser_;
}

template <typename T, typename D, typename WT, typename TT, typename TC>
void Bm3dDenoisingInvokerStep2<T, D, WT, TT, TC>::operator() (const Range& range) const
{
    const int size = (range.size() + 2 * borderSize_) * srcExtended_.cols;
    std::vector<WT> weightedSum(size, 0.0);
    std::vector<WT> weights(size, 0.0);
    int row_from = range.start;
    int row_to = range.end - 1;

    // Local vars for faster processing
    const int blockSize = templateWindowSize_;
    const int blockSizeSq = templateWindowSizeSq_;
    const int halfBlockSize = halfTemplateWindowSize_;
    const int searchWindowSize = searchWindowSize_;
    const int searchWindowSizeSq = searchWindowSizeSq_;
    const TT halfSearchWindowSize = (TT)halfSearchWindowSize_;
    const int hBM = hBM_;
    const int groupSize = groupSize_;

    const int step = srcExtended_.cols;
    const int dstStep = srcExtended_.cols;
    const int weiStep = srcExtended_.cols;
    const int dstcstep = dstStep - blockSize;
    const int weicstep = weiStep - blockSize;

    // Buffer to store 3D group
    BlockMatch<TT, int, TT> *bmBasic = new BlockMatch<TT, int, TT>[searchWindowSizeSq];
    BlockMatch<TT, int, TT> *bmSrc = new BlockMatch<TT, int, TT>[searchWindowSizeSq];
    for (int i = 0; i < searchWindowSizeSq; ++i)
    {
        bmBasic[i].init(blockSizeSq);
        bmSrc[i].init(blockSizeSq);
    }

    // First element in a group is always the reference patch. Hence distance is 0.
    bmBasic[0](0, halfSearchWindowSize, halfSearchWindowSize);
    bmSrc[0](0, halfSearchWindowSize, halfSearchWindowSize);

    // Sums of columns and rows for current pixel
    Array2d<int> distSums(searchWindowSize, searchWindowSize);

    // Sums of columns for current pixel (for lazy calc optimization)
    Array3d<int> colDistSums(blockSize, searchWindowSize, searchWindowSize);

    // Last elements of column sum (for each element in a row)
    Array3d<int> lastColDistSums(src_.cols, searchWindowSize, searchWindowSize);

    int firstColNum = -1;
    for (int j = row_from, jj = 0; j <= row_to; j += slidingStep_, jj += slidingStep_)
    {
        for (int i = 0; i < src_.cols; i += slidingStep_)
        {
            const T *currentPixelSrc = srcExtended_.ptr<T>(0) + step*j + i;
            const T *currentPixelBasic = basicExtended_.ptr<T>(0) + step*j + i;

            int elementSize = 1;

            // Calculate distSums using moving average filter approach.
            if (i == 0)
            {
                // Calculate distSums for the first element in a row
                calcDistSumsForFirstElementInRow(j, distSums, colDistSums, lastColDistSums, bmBasic, elementSize);
                firstColNum = 0;
            }
            else
            {
                if (j == row_from)
                {
                    // Calculate distSums for all elements in the first row
                    calcDistSumsForAllElementsInFirstRow(
                        j, i, firstColNum, distSums, colDistSums, lastColDistSums, bmBasic, elementSize);
                }
                else
                {
                    const int start_bx = blockSize + i - 1;
                    const int start_by = j - 1;
                    const int ax = halfSearchWindowSize + start_bx;
                    const int ay = halfSearchWindowSize + start_by;

                    const T a_up = basicExtended_.at<T>(ay, ax);
                    const T a_down = basicExtended_.at<T>(ay + blockSize, ax);

                    for (TT y = 0; y < searchWindowSize; y++)
                    {
                        int *distSumsRow = distSums.row_ptr(y);
                        int *colDistSumsRow = colDistSums.row_ptr(firstColNum, y);
                        int *lastColDistSumsRow = lastColDistSums.row_ptr(i, y);

                        const T *b_up_ptr = basicExtended_.ptr<T>(start_by + y);
                        const T *b_down_ptr = basicExtended_.ptr<T>(start_by + y + blockSize);

                        for (TT x = 0; x < searchWindowSize; x++)
                        {
                            // Remove from current pixel sum column sum with index "firstColNum"
                            distSumsRow[x] -= colDistSumsRow[x];

                            const int bx = start_bx + x;
                            colDistSumsRow[x] = lastColDistSumsRow[x] +
                                D::template calcUpDownDist<T>(a_up, a_down, b_up_ptr[bx], b_down_ptr[bx]);

                            distSumsRow[x] += colDistSumsRow[x];
                            lastColDistSumsRow[x] = colDistSumsRow[x];

                            if (x == halfSearchWindowSize && y == halfSearchWindowSize)
                                continue;

                            // Save the distance, coordinate and increase the counter
                            if (distSumsRow[x] < hBM)
                                bmBasic[elementSize++](distSumsRow[x], x, y);
                        }
                    }
                }

                firstColNum = (firstColNum + 1) % blockSize;
            }

            // Sort bmBasic by distance (first element is already sorted)
            std::sort(bmBasic + 1, bmBasic + elementSize);

            // Find the nearest power of 2 and cap the group size from the top
            elementSize = getLargestPowerOf2SmallerThan(elementSize);
            if (elementSize > groupSize)
                elementSize = groupSize;

            // Transform 2D patches
            for (int n = 0; n < elementSize; ++n)
            {
                const T *candidatePatchSrc = currentPixelSrc + step * bmBasic[n].coord_y + bmBasic[n].coord_x;
                const T *candidatePatchBasic = currentPixelBasic + step * bmBasic[n].coord_y + bmBasic[n].coord_x;
                TC::forwardTransform2D(candidatePatchSrc, bmSrc[n].data(), step, blockSize);
                TC::forwardTransform2D(candidatePatchBasic, bmBasic[n].data(), step, blockSize);
            }

            // Transform and shrink 1D columns
            int wienerCoefficients = 0;
            TT *thrMapPtr1D = thrMap_ + (elementSize - 1) * blockSizeSq;
            switch (elementSize)
            {
            case 16:
                for (int n = 0; n < blockSizeSq; n++)
                {
                    TC::forwardTransform16(bmSrc, n);
                    TC::forwardTransform16(bmBasic, n);
                    wienerCoefficients += WienerFiltering<16>(bmSrc, bmBasic, n, thrMapPtr1D);
                    TC::inverseTransform16(bmBasic, n);
                }
                break;
            case 8:
                for (int n = 0; n < blockSizeSq; n++)
                {
                    TC::forwardTransform8(bmSrc, n);
                    TC::forwardTransform8(bmBasic, n);
                    wienerCoefficients += WienerFiltering<8>(bmSrc, bmBasic, n, thrMapPtr1D);
                    TC::inverseTransform8(bmBasic, n);
                }
                break;
            case 4:
                for (int n = 0; n < blockSizeSq; n++)
                {
                    TC::forwardTransform4(bmSrc, n);
                    TC::forwardTransform4(bmBasic, n);
                    wienerCoefficients += WienerFiltering<4>(bmSrc, bmBasic, n, thrMapPtr1D);
                    TC::inverseTransform4(bmBasic, n);
                }
                break;
            case 2:
                for (int n = 0; n < blockSizeSq; n++)
                {
                    TC::forwardTransform2(bmSrc, n);
                    TC::forwardTransform2(bmBasic, n);
                    wienerCoefficients += WienerFiltering<2>(bmSrc, bmBasic, n, thrMapPtr1D);
                    TC::inverseTransform2(bmBasic, n);
                }
                break;
            case 1:
            {
                for (int n = 0; n < blockSizeSq; n++)
                    wienerCoefficients += WienerFiltering<1>(bmSrc, bmBasic, n, thrMapPtr1D);
            }
            break;
            default:
                for (int n = 0; n < blockSizeSq; n++)
                {
                    TC::forwardTransformN(bmSrc, n, elementSize);
                    TC::forwardTransformN(bmBasic, n, elementSize);
                    wienerCoefficients += WienerFiltering(bmSrc, bmBasic, n, thrMapPtr1D, elementSize);
                    TC::inverseTransformN(bmBasic, n, elementSize);
                }
            }

            // Inverse 2D transform
            for (int n = 0; n < elementSize; ++n)
                TC::inverseTransform2D(bmBasic[n].data(), blockSize);

            // Aggregate the results (increase sumNonZero to avoid division by zero)
            float weight = 1.0f / (float)(++wienerCoefficients);

            // Scale weight by element size
            weight *= elementSize;
            weight /= groupSize;

            // Put patches back to their original positions
            WT *dstPtr = weightedSum.data() + jj * dstStep + i;
            WT *weiPtr = weights.data() + jj * dstStep + i;
            const float *kaiser = kaiser_;

            for (int l = 0; l < elementSize; ++l)
            {
                const TT *block = bmBasic[l].data();
                int offset = bmBasic[l].coord_y * dstStep + bmBasic[l].coord_x;
                WT *d = dstPtr + offset;
                WT *dw = weiPtr + offset;

                for (int n = 0; n < blockSize; ++n)
                {
                    for (int m = 0; m < blockSize; ++m)
                    {
                        unsigned idx = n * blockSize + m;
                        *d += kaiser[idx] * block[idx] * weight;
                        *dw += kaiser[idx] * weight;
                        ++d, ++dw;
                    }
                    d += dstcstep;
                    dw += weicstep;
                }
            }
        } // i
    } // j

    // Cleanup
    for (int i = 0; i < searchWindowSizeSq; ++i)
    {
        bmBasic[i].release();
        bmSrc[i].release();
    }

    delete[] bmSrc;
    delete[] bmBasic;

    // Divide accumulation buffer by the corresponding weights
    for (int i = row_from, ii = 0; i <= row_to; ++i, ++ii)
    {
        T *d = dst_.ptr<T>(i);
        float *dE = weightedSum.data() + (ii + halfSearchWindowSize + halfBlockSize) * dstStep + halfSearchWindowSize;
        float *dw = weights.data() + (ii + halfSearchWindowSize + halfBlockSize) * dstStep + halfSearchWindowSize;
        for (int j = 0; j < dst_.cols; ++j)
            d[j] = cv::saturate_cast<T>(dE[j + halfBlockSize] / dw[j + halfBlockSize]);
    }
}


template <typename T, typename D, typename WT, typename TT, typename TC>
inline void Bm3dDenoisingInvokerStep2<T, D, WT, TT, TC>::calcDistSumsForFirstElementInRow(
    int i,
    Array2d<int>& distSums,
    Array3d<int>& colDistSums,
    Array3d<int>& lastColDistSums,
    BlockMatch<TT, int, TT> *bm,
    int &elementSize) const
{
    int j = 0;
    const int hBM = hBM_;
    const int blockSize = templateWindowSize_;
    const int searchWindowSize = searchWindowSize_;
    const TT halfSearchWindowSize = (TT)halfSearchWindowSize_;
    const int ay = halfSearchWindowSize + i;
    const int ax = halfSearchWindowSize + j;

    for (TT y = 0; y < searchWindowSize; ++y)
    {
        for (TT x = 0; x < searchWindowSize; ++x)
        {
            // Zeroize arrays
            distSums[y][x] = 0;
            for (int tx = 0; tx < blockSize; tx++)
                colDistSums[tx][y][x] = 0;

            int start_y = i + y;
            int start_x = j + x;

            for (int ty = 0; ty < blockSize; ty++)
                for (int tx = 0; tx < blockSize; tx++)
                {
                    int dist = D::template calcDist<T>(
                        basicExtended_,
                        ay + ty,
                        ax + tx,
                        start_y + ty,
                        start_x + tx);

                    distSums[y][x] += dist;
                    colDistSums[tx][y][x] += dist;
                }

            lastColDistSums[j][y][x] = colDistSums[blockSize - 1][y][x];

            if (x == halfSearchWindowSize && y == halfSearchWindowSize)
                continue;

            if (distSums[y][x] < hBM)
                bm[elementSize++](distSums[y][x], x, y);
        }
    }
}

template <typename T, typename D, typename WT, typename TT, typename TC>
inline void Bm3dDenoisingInvokerStep2<T, D, WT, TT, TC>::calcDistSumsForAllElementsInFirstRow(
    int i,
    int j,
    int firstColNum,
    Array2d<int>& distSums,
    Array3d<int>& colDistSums,
    Array3d<int>& lastColDistSums,
    BlockMatch<TT, int, TT> *bm,
    int &elementSize) const
{
    const int hBM = hBM_;
    const int blockSize = templateWindowSize_;
    const int searchWindowSize = searchWindowSize_;
    const TT halfSearchWindowSize = (TT)halfSearchWindowSize_;

    const int bx_start = blockSize - 1 + j;
    const int ax = halfSearchWindowSize + bx_start;
    const int ay = halfSearchWindowSize + i;

    for (TT y = 0; y < searchWindowSize; ++y)
    {
        for (TT x = 0; x < searchWindowSize; ++x)
        {
            distSums[y][x] -= colDistSums[firstColNum][y][x];

            colDistSums[firstColNum][y][x] = 0;
            int by = i + y;
            int bx = bx_start + x;

            for (int ty = 0; ty < blockSize; ty++)
                colDistSums[firstColNum][y][x] += D::template calcDist<T>(
                    basicExtended_,
                    ay + ty,
                    ax,
                    by + ty,
                    bx);

            distSums[y][x] += colDistSums[firstColNum][y][x];
            lastColDistSums[j][y][x] = colDistSums[firstColNum][y][x];

            if (x == halfSearchWindowSize && y == halfSearchWindowSize)
                continue;

            if (distSums[y][x] < hBM)
                bm[elementSize++](distSums[y][x], x, y);
        }
    }
}

}  // namespace xphoto
}  // namespace cv

#endif
