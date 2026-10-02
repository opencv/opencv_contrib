//M*//////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

/****************************************************************************************\
*    Very fast SAD-based (Sum-of-Absolute-Diffrences) stereo correspondence algorithm.   *
*    Contributed by Kurt Konolige                                                        *
\****************************************************************************************/

#include "precomp.hpp"
#include <stdio.h>
#include <limits>

namespace cv
{
    namespace stereo
    {

        struct StereoBinaryBMParams
        {
            StereoBinaryBMParams(int _numDisparities = 64, int _kernelSize = 9)
            {
                preFilterType = StereoBinaryBM::PREFILTER_XSOBEL;
                preFilterSize = 9;
                preFilterCap = 31;
                kernelSize = _kernelSize;
                minDisparity = 0;
                numDisparities = _numDisparities > 0 ? _numDisparities : 64;
                textureThreshold = 10;
                uniquenessRatio = 15;
                speckleRange = speckleWindowSize = 0;
                disp12MaxDiff = -1;
                dispType = CV_16S;
                usePrefilter = false;
                regionRemoval = 1;
                scalling = 4;
                kernelType = CV_MODIFIED_CENSUS_TRANSFORM;
                agregationWindowSize = 9;
            }

            int preFilterType;
            int preFilterSize;
            int preFilterCap;
            int kernelSize;
            int minDisparity;
            int numDisparities;
            int textureThreshold;
            int uniquenessRatio;
            int speckleRange;
            int speckleWindowSize;
            int disp12MaxDiff;
            int dispType;
            int scalling;
            bool usePrefilter;
            int regionRemoval;
            int kernelType;
            int agregationWindowSize;
        };

        static void prefilterNorm(const Mat& src, Mat& dst, int winsize, int ftzero, uchar* buf)
        {
            int x, y, wsz2 = winsize / 2;
            int* vsum = (int*)alignPtr(buf + (wsz2 + 1)*sizeof(vsum[0]), 32);
            int scale_g = winsize*winsize / 8, scale_s = (1024 + scale_g) / (scale_g * 2);
            const int OFS = 256 * 5, TABSZ = OFS * 2 + 256;
            uchar tab[TABSZ];
            const uchar* sptr = src.ptr();
            int srcstep = (int)src.step;
            Size size = src.size();

            scale_g *= scale_s;

            for (x = 0; x < TABSZ; x++)
                tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero * 2 : x - OFS + ftzero);

            for (x = 0; x < size.width; x++)
                vsum[x] = (ushort)(sptr[x] * (wsz2 + 2));
            for (y = 1; y < wsz2; y++)
            {
                for (x = 0; x < size.width; x++)
                    vsum[x] = (ushort)(vsum[x] + sptr[srcstep*y + x]);
            }
            for (y = 0; y < size.height; y++)
            {
                const uchar* top = sptr + srcstep*MAX(y - wsz2 - 1, 0);
                const uchar* bottom = sptr + srcstep*MIN(y + wsz2, size.height - 1);
                const uchar* prev = sptr + srcstep*MAX(y - 1, 0);
                const uchar* curr = sptr + srcstep*y;
                const uchar* next = sptr + srcstep*MIN(y + 1, size.height - 1);
                uchar* dptr = dst.ptr<uchar>(y);
                for (x = 0; x < size.width; x++)
                    vsum[x] = (ushort)(vsum[x] + bottom[x] - top[x]);

                for (x = 0; x <= wsz2; x++)
                {
                    vsum[-x - 1] = vsum[0];
                    vsum[size.width + x] = vsum[size.width - 1];
                }

                int sum = vsum[0] * (wsz2 + 1);
                for (x = 1; x <= wsz2; x++)
                    sum += vsum[x];
                int val = ((curr[0] * 5 + curr[1] + prev[0] + next[0])*scale_g - sum*scale_s) >> 10;
                dptr[0] = tab[val + OFS];
                for (x = 1; x < size.width - 1; x++)
                {
                    sum += vsum[x + wsz2] - vsum[x - wsz2 - 1];
                    val = ((curr[x] * 4 + curr[x - 1] + curr[x + 1] + prev[x] + next[x])*scale_g - sum*scale_s) >> 10;
                    dptr[x] = tab[val + OFS];
                }

                sum += vsum[x + wsz2] - vsum[x - wsz2 - 1];
                val = ((curr[x] * 5 + curr[x - 1] + prev[x] + next[x])*scale_g - sum*scale_s) >> 10;
                dptr[x] = tab[val + OFS];
            }
        }

        static void
            prefilterXSobel(const Mat& src, Mat& dst, int ftzero)
        {
            int x, y;
            const int OFS = 256 * 4, TABSZ = OFS * 2 + 256;
            uchar tab[TABSZ];
            Size size = src.size();

            for (x = 0; x < TABSZ; x++)
                tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero * 2 : x - OFS + ftzero);
            uchar val0 = tab[0 + OFS];

#if CV_SSE2
            volatile bool useSIMD = checkHardwareSupport(CV_CPU_SSE2);
#endif

            for (y = 0; y < size.height - 1; y += 2)
            {
                const uchar* srow1 = src.ptr<uchar>(y);
                const uchar* srow0 = y > 0 ? srow1 - src.step : size.height > 1 ? srow1 + src.step : srow1;
                const uchar* srow2 = y < size.height - 1 ? srow1 + src.step : size.height > 1 ? srow1 - src.step : srow1;
                const uchar* srow3 = y < size.height - 2 ? srow1 + src.step * 2 : srow1;
                uchar* dptr0 = dst.ptr<uchar>(y);
                uchar* dptr1 = dptr0 + dst.step;

                dptr0[0] = dptr0[size.width - 1] = dptr1[0] = dptr1[size.width - 1] = val0;
                x = 1;

#if CV_SSE2
                if (useSIMD)
                {
                    __m128i z = _mm_setzero_si128(), ftz = _mm_set1_epi16((short)ftzero),
                        ftz2 = _mm_set1_epi8(cv::saturate_cast<uchar>(ftzero * 2));
                    for (; x <= size.width - 9; x += 8)
                    {
                        __m128i c0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow0 + x - 1)), z);
                        __m128i c1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow1 + x - 1)), z);
                        __m128i d0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow0 + x + 1)), z);
                        __m128i d1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow1 + x + 1)), z);

                        d0 = _mm_sub_epi16(d0, c0);
                        d1 = _mm_sub_epi16(d1, c1);

                        __m128i c2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow2 + x - 1)), z);
                        __m128i c3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow3 + x - 1)), z);
                        __m128i d2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow2 + x + 1)), z);
                        __m128i d3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow3 + x + 1)), z);

                        d2 = _mm_sub_epi16(d2, c2);
                        d3 = _mm_sub_epi16(d3, c3);

                        __m128i v0 = _mm_add_epi16(d0, _mm_add_epi16(d2, _mm_add_epi16(d1, d1)));
                        __m128i v1 = _mm_add_epi16(d1, _mm_add_epi16(d3, _mm_add_epi16(d2, d2)));
                        v0 = _mm_packus_epi16(_mm_add_epi16(v0, ftz), _mm_add_epi16(v1, ftz));
                        v0 = _mm_min_epu8(v0, ftz2);

                        _mm_storel_epi64((__m128i*)(dptr0 + x), v0);
                        _mm_storel_epi64((__m128i*)(dptr1 + x), _mm_unpackhi_epi64(v0, v0));
                    }
                }
#endif

                for (; x < size.width - 1; x++)
                {
                    int d0 = srow0[x + 1] - srow0[x - 1], d1 = srow1[x + 1] - srow1[x - 1],
                        d2 = srow2[x + 1] - srow2[x - 1], d3 = srow3[x + 1] - srow3[x - 1];
                    int v0 = tab[d0 + d1 * 2 + d2 + OFS];
                    int v1 = tab[d1 + d2 * 2 + d3 + OFS];
                    dptr0[x] = (uchar)v0;
                    dptr1[x] = (uchar)v1;
                }
            }

            for (; y < size.height; y++)
            {
                uchar* dptr = dst.ptr<uchar>(y);
                for (x = 0; x < size.width; x++)
                    dptr[x] = val0;
            }
        }

        static const int DISPARITY_SHIFT = 4;

        struct PrefilterInvoker : public ParallelLoopBody
        {
            PrefilterInvoker(const Mat& left0, const Mat& right0, Mat& left, Mat& right,
                uchar* buf0, uchar* buf1, StereoBinaryBMParams* _state)
            {
                imgs0[0] = &left0; imgs0[1] = &right0;
                imgs[0] = &left; imgs[1] = &right;
                buf[0] = buf0; buf[1] = buf1;
                state = _state;
            }

            void operator()(const Range& range) const CV_OVERRIDE
            {
                for (int i = range.start; i < range.end; i++)
                {
                    if (state->preFilterType == StereoBinaryBM::PREFILTER_NORMALIZED_RESPONSE)
                        prefilterNorm(*imgs0[i], *imgs[i], state->preFilterSize, state->preFilterCap, buf[i]);
                    else
                        prefilterXSobel(*imgs0[i], *imgs[i], state->preFilterCap);
                }
            }

            const Mat* imgs0[2];
            Mat* imgs[2];
            uchar* buf[2];
            StereoBinaryBMParams* state;
        };

        class StereoBinaryBMImpl CV_FINAL : public StereoBinaryBM, public Matching
        {
        public:
            StereoBinaryBMImpl(): Matching(64)
            {
                params = StereoBinaryBMParams();
            }

            StereoBinaryBMImpl(int _numDisparities, int _kernelSize) : Matching(_numDisparities)
            {
                params = StereoBinaryBMParams(_numDisparities, _kernelSize);
            }

            void compute(InputArray leftarr, InputArray rightarr, OutputArray disparr) CV_OVERRIDE
            {
                int dtype = disparr.fixedType() ? disparr.type() : params.dispType;
                Size leftsize = leftarr.size();

                if (leftarr.size() != rightarr.size())
                    CV_Error(Error::StsUnmatchedSizes, "All the images must have the same size");

                if (leftarr.type() != CV_8UC1 || rightarr.type() != CV_8UC1)
                    CV_Error(Error::StsUnsupportedFormat, "Both input images must have CV_8UC1");

                if (dtype != CV_16SC1 && dtype != CV_32FC1)
                    CV_Error(Error::StsUnsupportedFormat, "Disparity image must have CV_16SC1 or CV_32FC1 format");

                if (params.preFilterType != PREFILTER_NORMALIZED_RESPONSE &&
                    params.preFilterType != PREFILTER_XSOBEL)
                    CV_Error(Error::StsOutOfRange, "preFilterType must be = CV_STEREO_BM_NORMALIZED_RESPONSE");

                if (params.preFilterSize < 5 || params.preFilterSize > 255 || params.preFilterSize % 2 == 0)
                    CV_Error(Error::StsOutOfRange, "preFilterSize must be odd and be within 5..255");

                if (params.preFilterCap < 1 || params.preFilterCap > 63)
                    CV_Error(Error::StsOutOfRange, "preFilterCap must be within 1..63");

                if (params.kernelSize < 5 || params.kernelSize > 255 || params.kernelSize % 2 == 0 ||
                    params.kernelSize >= std::min(leftsize.width, leftsize.height))
                    CV_Error(Error::StsOutOfRange, "kernelSize must be odd, be within 5..255 and be not larger than image width or height");

                if (params.numDisparities <= 0 || params.numDisparities % 16 != 0)
                    CV_Error(Error::StsOutOfRange, "numDisparities must be positive and divisble by 16");

                if (params.textureThreshold < 0)
                    CV_Error(Error::StsOutOfRange, "texture threshold must be non-negative");

                if (params.uniquenessRatio < 0)
                    CV_Error(Error::StsOutOfRange, "uniqueness ratio must be non-negative");

                int FILTERED = (params.minDisparity - 1) << DISPARITY_SHIFT;

                Mat left0 = leftarr.getMat(), right0 = rightarr.getMat();
                Mat disp0 = disparr.getMat();

                int width = left0.cols;
                int height = left0.rows;
                if (puss.total() != (size_t)width * height)
                {
                    speckleX.create(height, width);
                    speckleY.create(height, width);
                    puss.create(height, width);

                    censusImage[0].create(left0.rows,left0.cols,CV_32SC4);
                    censusImage[1].create(left0.rows,left0.cols,CV_32SC4);

                    partialSumsLR.create(left0.rows + 1,(left0.cols + 1) * (params.numDisparities + 1),CV_16S);
                    agregatedHammingLRCost.create(left0.rows + 1,(left0.cols + 1) * (params.numDisparities + 1),CV_16S);
                    hammingDistance.create(left0.rows, left0.cols * (params.numDisparities + 1),CV_16S);

                    preFilteredImg0.create(left0.size(), CV_8U);
                    preFilteredImg1.create(left0.size(), CV_8U);

                    aux.create(height,width,CV_8UC1);
                }

                Mat left = preFilteredImg0, right = preFilteredImg1;
                int bufSize1 = (int)((width + params.preFilterSize + 2) * sizeof(int) + 256);
                if(params.usePrefilter == true)
                {
                    uchar *_buf = slidingSumBuf.ptr();

                    parallel_for_(Range(0, 2), PrefilterInvoker(left0, right0, left, right, _buf, _buf + bufSize1, &params), 1);
                }
                else if(params.usePrefilter == false)
                {
                    left = left0;
                    right = right0;
                }
                if(params.kernelType == CV_SPARSE_CENSUS)
                {
                    censusTransform(left,right,params.kernelSize,censusImage[0],censusImage[1],CV_SPARSE_CENSUS);
                }
                else if(params.kernelType == CV_DENSE_CENSUS)
                {
                    censusTransform(left,right,params.kernelSize,censusImage[0],censusImage[1],CV_DENSE_CENSUS);
                }
                else if(params.kernelType == CV_CS_CENSUS)
                {
                    symetricCensusTransform(left,right,params.kernelSize,censusImage[0],censusImage[1],CV_CS_CENSUS);
                }
                else if(params.kernelType == CV_MODIFIED_CS_CENSUS)
                {
                    symetricCensusTransform(left,right,params.kernelSize,censusImage[0],censusImage[1],CV_MODIFIED_CS_CENSUS);
                }
                else if(params.kernelType == CV_MODIFIED_CENSUS_TRANSFORM)
                {
                    modifiedCensusTransform(left,right,params.kernelSize,censusImage[0],censusImage[1],CV_MODIFIED_CENSUS_TRANSFORM,0);
                }
                else if(params.kernelType == CV_MEAN_VARIATION)
                {
                    Mat blurLeft; blur(left, blurLeft, Size(params.kernelSize, params.kernelSize));
                    Mat blurRight; blur(right, blurRight, Size(params.kernelSize, params.kernelSize));
                    modifiedCensusTransform(left, right, params.kernelSize, censusImage[0], censusImage[1], CV_MEAN_VARIATION, 0,
                            blurLeft, blurRight);
                }
                else if(params.kernelType == CV_STAR_KERNEL)
                {
                    starCensusTransform(left,right,params.kernelSize,censusImage[0],censusImage[1]);
                }
                hammingDistanceBlockMatching(censusImage[0], censusImage[1], hammingDistance, params.kernelSize);
                costGathering(hammingDistance, partialSumsLR);
                blockAgregation(partialSumsLR, params.agregationWindowSize, agregatedHammingLRCost);
                dispartyMapFormation(agregatedHammingLRCost, disp0, 3);
                Median1x9Filter<uint8_t>(disp0, aux);
                Median9x1Filter<uint8_t>(aux,disp0);

                if(params.regionRemoval == CV_SPECKLE_REMOVAL_AVG_ALGORITHM)
                {
                    smallRegionRemoval<uint8_t>(disp0.clone(),params.speckleWindowSize,disp0);
                }
                else if(params.regionRemoval == CV_SPECKLE_REMOVAL_ALGORITHM)
                {
                    if (params.speckleRange >= 0 && params.speckleWindowSize > 0)
                        filterSpeckles(disp0, FILTERED, params.speckleWindowSize, params.speckleRange, slidingSumBuf);
                }
            }
            int getAgregationWindowSize() const CV_OVERRIDE { return params.agregationWindowSize;}
            void setAgregationWindowSize(int value = 9) CV_OVERRIDE { CV_Assert(value % 2 != 0); params.agregationWindowSize = value;}

            int getBinaryKernelType() const CV_OVERRIDE { return params.kernelType;}
            void setBinaryKernelType(int value = CV_MODIFIED_CENSUS_TRANSFORM) CV_OVERRIDE { CV_Assert(value < 7); params.kernelType = value; }

            int getSpekleRemovalTechnique() const CV_OVERRIDE { return params.regionRemoval;}
            void setSpekleRemovalTechnique(int factor = CV_SPECKLE_REMOVAL_AVG_ALGORITHM) CV_OVERRIDE { CV_Assert(factor < 2); params.regionRemoval = factor; }

            bool getUsePrefilter() const CV_OVERRIDE { return params.usePrefilter;}
            void setUsePrefilter(bool value = false) CV_OVERRIDE { params.usePrefilter = value;}

            int getScalleFactor() const CV_OVERRIDE { return params.scalling;}
            void setScalleFactor(int factor = 4) CV_OVERRIDE { CV_Assert(factor > 0); params.scalling = factor; setScallingFactor(factor); }

            int getMinDisparity() const CV_OVERRIDE { return params.minDisparity; }
            void setMinDisparity(int minDisparity) CV_OVERRIDE { CV_Assert(minDisparity >= 0); params.minDisparity = minDisparity; }

            int getNumDisparities() const CV_OVERRIDE { return params.numDisparities; }
            void setNumDisparities(int numDisparities) CV_OVERRIDE { CV_Assert(numDisparities > 0); params.numDisparities = numDisparities; }

            int getBlockSize() const CV_OVERRIDE { return params.kernelSize; }
            void setBlockSize(int blockSize) CV_OVERRIDE { CV_Assert(blockSize % 2 != 0); params.kernelSize = blockSize; }

            int getSpeckleWindowSize() const CV_OVERRIDE { return params.speckleWindowSize; }
            void setSpeckleWindowSize(int speckleWindowSize) CV_OVERRIDE { CV_Assert(speckleWindowSize >= 0); params.speckleWindowSize = speckleWindowSize; }

            int getSpeckleRange() const CV_OVERRIDE { return params.speckleRange; }
            void setSpeckleRange(int speckleRange) CV_OVERRIDE { CV_Assert(speckleRange >= 0); params.speckleRange = speckleRange; }

            int getDisp12MaxDiff() const CV_OVERRIDE { return params.disp12MaxDiff; }
            void setDisp12MaxDiff(int disp12MaxDiff) CV_OVERRIDE { CV_Assert(disp12MaxDiff >= 0); params.disp12MaxDiff = disp12MaxDiff; }

            int getPreFilterType() const CV_OVERRIDE { return params.preFilterType; }
            void setPreFilterType(int preFilterType) CV_OVERRIDE { CV_Assert(preFilterType >= 0); params.preFilterType = preFilterType; }

            int getPreFilterSize() const CV_OVERRIDE { return params.preFilterSize; }
            void setPreFilterSize(int preFilterSize) CV_OVERRIDE { CV_Assert(preFilterSize >= 0);  params.preFilterSize = preFilterSize; }

            int getPreFilterCap() const CV_OVERRIDE { return params.preFilterCap; }
            void setPreFilterCap(int preFilterCap) CV_OVERRIDE { CV_Assert(preFilterCap >= 0); params.preFilterCap = preFilterCap; }

            int getTextureThreshold() const CV_OVERRIDE { return params.textureThreshold; }
            void setTextureThreshold(int textureThreshold) CV_OVERRIDE { CV_Assert(textureThreshold >= 0); params.textureThreshold = textureThreshold; }

            int getUniquenessRatio() const CV_OVERRIDE { return params.uniquenessRatio; }
            void setUniquenessRatio(int uniquenessRatio) CV_OVERRIDE { CV_Assert(uniquenessRatio >= 0); params.uniquenessRatio = uniquenessRatio; }

            int getSmallerBlockSize() const CV_OVERRIDE { return 0; }
            void setSmallerBlockSize(int) CV_OVERRIDE {}

            void write(FileStorage& fs) const CV_OVERRIDE
            {
                fs << "name" << name_
                    << "minDisparity" << params.minDisparity
                    << "numDisparities" << params.numDisparities
                    << "blockSize" << params.kernelSize
                    << "speckleWindowSize" << params.speckleWindowSize
                    << "speckleRange" << params.speckleRange
                    << "disp12MaxDiff" << params.disp12MaxDiff
                    << "preFilterType" << params.preFilterType
                    << "preFilterSize" << params.preFilterSize
                    << "preFilterCap" << params.preFilterCap
                    << "textureThreshold" << params.textureThreshold
                    << "uniquenessRatio" << params.uniquenessRatio;
            }

            void read(const FileNode& fn) CV_OVERRIDE
            {
                FileNode n = fn["name"];
                CV_Assert(n.isString() && String(n) == name_);
                params.minDisparity = (int)fn["minDisparity"];
                params.numDisparities = (int)fn["numDisparities"];
                params.kernelSize = (int)fn["blockSize"];
                params.speckleWindowSize = (int)fn["speckleWindowSize"];
                params.speckleRange = (int)fn["speckleRange"];
                params.disp12MaxDiff = (int)fn["disp12MaxDiff"];
                params.preFilterType = (int)fn["preFilterType"];
                params.preFilterSize = (int)fn["preFilterSize"];
                params.preFilterCap = (int)fn["preFilterCap"];
                params.textureThreshold = (int)fn["textureThreshold"];
                params.uniquenessRatio = (int)fn["uniquenessRatio"];
            }

            StereoBinaryBMParams params;
            Mat preFilteredImg0, preFilteredImg1, cost, dispbuf;
            Mat slidingSumBuf;
            Mat censusImage[2];
            Mat hammingDistance;
            Mat partialSumsLR;
            Mat agregatedHammingLRCost;
            Mat aux;
            static const char* name_;
        };

        const char* StereoBinaryBMImpl::name_ = "StereoBinaryMatcher.BM";

        Ptr<StereoBinaryBM> StereoBinaryBM::create(int _numDisparities, int _kernelSize)
        {
            return makePtr<StereoBinaryBMImpl>(_numDisparities, _kernelSize);
        }
    }
}
/* End of file. */
