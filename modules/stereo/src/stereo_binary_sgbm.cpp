/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
/*
This is a variation of
"Stereo Processing by Semiglobal Matching and Mutual Information"
by Heiko Hirschmuller.
We match blocks rather than individual pixels, thus the algorithm is called
SGBM (Semi-global block matching)
*/

#include "precomp.hpp"
#include <limits.h>

namespace cv
{
    namespace stereo
    {
        typedef uchar PixType;
        typedef short CostType;
        typedef short DispType;
        enum { NR = 16, NR2 = NR/2 };

        struct StereoBinarySGBMParams
        {
            StereoBinarySGBMParams()
            {
                minDisparity = numDisparities = 0;
                kernelSize = 0;
                P1 = P2 = 0;
                disp12MaxDiff = 0;
                preFilterCap = 0;
                uniquenessRatio = 0;
                speckleWindowSize = 0;
                speckleRange = 0;
                mode = StereoBinarySGBM::MODE_SGBM;
            }
            StereoBinarySGBMParams( int _minDisparity, int _numDisparities, int _SADWindowSize,
                int _P1, int _P2, int _disp12MaxDiff, int _preFilterCap,
                int _uniquenessRatio, int _speckleWindowSize, int _speckleRange,
                int _mode )
            {
                minDisparity = _minDisparity;
                numDisparities = _numDisparities;
                kernelSize = _SADWindowSize;
                P1 = _P1;
                P2 = _P2;
                disp12MaxDiff = _disp12MaxDiff;
                preFilterCap = _preFilterCap;
                uniquenessRatio = _uniquenessRatio;
                speckleWindowSize = _speckleWindowSize;
                speckleRange = _speckleRange;
                mode = _mode;
                regionRemoval = 1;
                kernelType = CV_MODIFIED_CENSUS_TRANSFORM;
                subpixelInterpolationMethod = CV_QUADRATIC_INTERPOLATION;
            }
            int minDisparity;
            int numDisparities;
            int kernelSize;
            int preFilterCap;
            int uniquenessRatio;
            int P1;
            int P2;
            int speckleWindowSize;
            int speckleRange;
            int disp12MaxDiff;
            int mode;
            int regionRemoval;
            int kernelType;
            int subpixelInterpolationMethod;
        };

        /*
        computes disparity for "roi" in img1 w.r.t. img2 and write it to disp1buf.
        that is, disp1buf(x, y)=d means that img1(x+roi.x, y+roi.y) ~ img2(x+roi.x-d, y+roi.y).
        minD <= d < maxD.
        disp2full is the reverse disparity map, that is:
        disp2full(x+roi.x,y+roi.y)=d means that img2(x+roi.x, y+roi.y) ~ img1(x+roi.x+d, y+roi.y)
        note that disp1buf will have the same size as the roi and
        disp2full will have the same size as img1 (or img2).
        On exit disp2buf is not the final disparity, it is an intermediate result that becomes
        final after all the tiles are processed.
        the disparity in disp1buf is written with sub-pixel accuracy
        (4 fractional bits, see StereoSGBM::DISP_SCALE),
        using quadratic interpolation, while the disparity in disp2buf
        is written as is, without interpolation.
        disp2cost also has the same size as img1 (or img2).
        It contains the minimum current cost, used to find the best disparity, corresponding to the minimal cost.
        */
        static void computeDisparityBinarySGBM( const Mat& img1, const Mat& img2,
            Mat& disp1, const StereoBinarySGBMParams& params,
            Mat& buffer,const Mat& hamDist)
        {
#if CV_SSE2
            static const uchar LSBTab[] =
            {
                0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0
            };

            volatile bool useSIMD = checkHardwareSupport(CV_CPU_SSE2);
#endif

            const int ALIGN = 16;
            const int DISP_SHIFT = StereoMatcher::DISP_SHIFT;
            const int DISP_SCALE = (1 << DISP_SHIFT);
            const CostType MAX_COST = SHRT_MAX;
            int minD = params.minDisparity, maxD = minD + params.numDisparities;
            Size kernelSize;
            kernelSize.width = kernelSize.height = params.kernelSize > 0 ? params.kernelSize : 5;
            int uniquenessRatio = params.uniquenessRatio >= 0 ? params.uniquenessRatio : 10;
            int disp12MaxDiff = params.disp12MaxDiff > 0 ? params.disp12MaxDiff : 1;
            int P1 = params.P1 > 0 ? params.P1 : 2, P2 = std::max(params.P2 > 0 ? params.P2 : 5, P1+1);
            int k, width = disp1.cols, height = disp1.rows;
            int minX1 = std::max(-maxD, 0), maxX1 = width + std::min(minD, 0);
            int D = maxD - minD, width1 = maxX1 - minX1;
            int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP*DISP_SCALE;
            int SW2 = kernelSize.width/2, SH2 = kernelSize.height/2;
            bool fullDP = params.mode == StereoBinarySGBM::MODE_HH;
            int npasses = fullDP ? 2 : 1;

            if( minX1 >= maxX1 )
            {
                disp1 = Scalar::all(INVALID_DISP_SCALED);
                return;
            }
            CV_Assert( D % 16 == 0 );
            // NR - the number of directions. the loop on x below that computes Lr assumes that NR == 8.
            // if you change NR, please, modify the loop as well.
            int D2 = D+16, NRD2 = NR2*D2;
            // the number of L_r(.,.) and min_k L_r(.,.) lines in the buffer:
            // for 8-way dynamic programming we need the current row and
            // the previous row, i.e. 2 rows in total
            const int NLR = 2;
            const int LrBorder = NLR - 1;
            int ww = img2.cols;
            short *ham;
            ham = (short *)hamDist.data;
            // for each possible stereo match (img1(x,y) <=> img2(x-d,y))
            // we keep pixel difference cost (C) and the summary cost over NR directions (S).
            // we also keep all the partial costs for the previous line L_r(x,d) and also min_k L_r(x, k)
            size_t costBufSize = width1*D;
            size_t CSBufSize = costBufSize*(fullDP ? height : 1);
            size_t minLrSize = (width1 + LrBorder*2)*NR2, LrSize = minLrSize*D2;
            int hsumBufNRows = SH2*2 + 2;
            size_t totalBufSize = (LrSize + minLrSize)*NLR*sizeof(CostType) + // minLr[] and Lr[]
                costBufSize*(hsumBufNRows + 1)*sizeof(CostType) + // hsumBuf, pixdiff
                CSBufSize*2*sizeof(CostType) + // C, S
                width*16*img1.channels()*sizeof(PixType) + // temp buffer for computing per-pixel cost
                width*(sizeof(CostType) + sizeof(DispType)) + 1024; // disp2cost + disp2
            if( buffer.empty() || !buffer.isContinuous() ||
                buffer.cols*buffer.rows*buffer.elemSize() < totalBufSize )
                buffer.create(1, (int)totalBufSize, CV_8U);
            // summary cost over different (nDirs) directions
            CostType* Cbuf = (CostType*)alignPtr(buffer.ptr(), ALIGN);
            CostType* Sbuf = Cbuf + CSBufSize;
            CostType* hsumBuf = Sbuf + CSBufSize;
            CostType* pixDiff = hsumBuf + costBufSize*hsumBufNRows;
            CostType* disp2cost = pixDiff + costBufSize + (LrSize + minLrSize)*NLR;
            DispType* disp2ptr = (DispType*)(disp2cost + width);
            //            PixType* tempBuf = (PixType*)(disp2ptr + width);
            // add P2 to every C(x,y). it saves a few operations in the inner loops
            for( k = 0; k < width1*D; k++ )
                Cbuf[k] = (CostType)P2;
            for( int pass = 1; pass <= npasses; pass++ )
            {
                int x1, y1, x2, y2, dx, dy;
                if( pass == 1 )
                {
                    y1 = 0; y2 = height; dy = 1;
                    x1 = 0; x2 = width1; dx = 1;
                }
                else
                {
                    y1 = height-1; y2 = -1; dy = -1;
                    x1 = width1-1; x2 = -1; dx = -1;
                }
                CostType *Lr[NLR]={0}, *minLr[NLR]={0};
                for( k = 0; k < NLR; k++ )
                {
                    // shift Lr[k] and minLr[k] pointers, because we allocated them with the borders,
                    // and will occasionally use negative indices with the arrays
                    // we need to shift Lr[k] pointers by 1, to give the space for d=-1.
                    // however, then the alignment will be imperfect, i.e. bad for SSE,
                    // thus we shift the pointers by 8 (8*sizeof(short) == 16 - ideal alignment)
                    Lr[k] = pixDiff + costBufSize + LrSize*k + NRD2*LrBorder + 8;
                    memset( Lr[k] - LrBorder*NRD2 - 8, 0, LrSize*sizeof(CostType) );
                    minLr[k] = pixDiff + costBufSize + LrSize*NLR + minLrSize*k + NR2*LrBorder;
                    memset( minLr[k] - LrBorder*NR2, 0, minLrSize*sizeof(CostType) );
                }
                for( int y = y1; y != y2; y += dy )
                {
                    int x, d;
                    DispType* disp1ptr = disp1.ptr<DispType>(y);
                    CostType* C = Cbuf + (!fullDP ? 0 : y*costBufSize);
                    CostType* S = Sbuf + (!fullDP ? 0 : y*costBufSize);
                    if( pass == 1 ) // compute C on the first pass, and reuse it on the second pass, if any.
                    {
                        int dy1 = y == 0 ? 0 : y + SH2, dy2 = y == 0 ? SH2 : dy1;
                        for( k = dy1; k <= dy2; k++ )
                        {
                            CostType* hsumAdd = hsumBuf + (std::min(k, height-1) % hsumBufNRows)*costBufSize;
                            if( k < height )
                            {
                                for(int ii = 0; ii <= ww; ii++)
                                {
                                    for(int dd = 0; dd <= params.numDisparities; dd++)
                                    {
                                        pixDiff[ii * (params.numDisparities)+ dd] = (CostType)(ham[(k * (ww) + ii) * (params.numDisparities +1) + dd]);
                                    }
                                }
                                memset(hsumAdd, 0, D*sizeof(CostType));
                                for( x = 0; x <= SW2*D; x += D )
                                {
                                    int scale = x == 0 ? SW2 + 1 : 1;
                                    for( d = 0; d < D; d++ )
                                        hsumAdd[d] = (CostType)(hsumAdd[d] + pixDiff[x + d]*scale);
                                }

                                if( y > 0 )
                                {
                                    const CostType* hsumSub = hsumBuf + (std::max(y - SH2 - 1, 0) % hsumBufNRows)*costBufSize;
                                    const CostType* Cprev = !fullDP || y == 0 ? C : C - costBufSize;

                                    for( x = D; x < width1*D; x += D )
                                    {
                                        const CostType* pixAdd = pixDiff + std::min(x + SW2*D, (width1-1)*D);
                                        const CostType* pixSub = pixDiff + std::max(x - (SW2+1)*D, 0);

#if CV_SSE2
                                        if( useSIMD )
                                        {
                                            for( d = 0; d < D; d += 8 )
                                            {
                                                __m128i hv = _mm_load_si128((const __m128i*)(hsumAdd + x - D + d));
                                                __m128i Cx = _mm_load_si128((__m128i*)(Cprev + x + d));
                                                hv = _mm_adds_epi16(_mm_subs_epi16(hv,
                                                    _mm_load_si128((const __m128i*)(pixSub + d))),
                                                    _mm_load_si128((const __m128i*)(pixAdd + d)));
                                                Cx = _mm_adds_epi16(_mm_subs_epi16(Cx,
                                                    _mm_load_si128((const __m128i*)(hsumSub + x + d))),
                                                    hv);
                                                _mm_store_si128((__m128i*)(hsumAdd + x + d), hv);
                                                _mm_store_si128((__m128i*)(C + x + d), Cx);
                                            }
                                        }
                                        else
#endif
                                        {
                                            for( d = 0; d < D; d++ )
                                            {
                                                int hv = hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
                                                C[x + d] = (CostType)(Cprev[x + d] + hv - hsumSub[x + d]);
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    for( x = D; x < width1*D; x += D )
                                    {
                                        const CostType* pixAdd = pixDiff + std::min(x + SW2*D, (width1-1)*D);
                                        const CostType* pixSub = pixDiff + std::max(x - (SW2+1)*D, 0);
                                        for( d = 0; d < D; d++ )
                                            hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
                                    }
                                }
                            }
                            if( y == 0 )
                            {
                                int scale = k == 0 ? SH2 + 1 : 1;
                                for( x = 0; x < width1*D; x++ )
                                    C[x] = (CostType)(C[x] + hsumAdd[x]*scale);
                            }
                        }
                        // also, clear the S buffer
                        for( k = 0; k < width1*D; k++ )
                            S[k] = 0;
                    }
                    // clear the left and the right borders
                    memset( Lr[0] - NRD2*LrBorder - 8, 0, NRD2*LrBorder*sizeof(CostType) );
                    memset( Lr[0] + width1*NRD2 - 8, 0, NRD2*LrBorder*sizeof(CostType) );
                    memset( minLr[0] - NR2*LrBorder, 0, NR2*LrBorder*sizeof(CostType) );
                    memset( minLr[0] + width1*NR2, 0, NR2*LrBorder*sizeof(CostType) );
                    /*
                    [formula 13 in the paper]
                    compute L_r(p, d) = C(p, d) +
                    min(L_r(p-r, d),
                    L_r(p-r, d-1) + P1,
                    L_r(p-r, d+1) + P1,
                    min_k L_r(p-r, k) + P2) - min_k L_r(p-r, k)
                    where p = (x,y), r is one of the directions.
                    we process all the directions at once:
                    0: r=(-dx, 0)
                    1: r=(-1, -dy)
                    2: r=(0, -dy)
                    3: r=(1, -dy)
                    4: r=(-2, -dy)
                    5: r=(-1, -dy*2)
                    6: r=(1, -dy*2)
                    7: r=(2, -dy)
                    */
                    for( x = x1; x != x2; x += dx )
                    {
                        int xm = x*NR2, xd = xm*D2;
                        int delta0 = minLr[0][xm - dx*NR2] + P2, delta1 = minLr[1][xm - NR2 + 1] + P2;
                        int delta2 = minLr[1][xm + 2] + P2, delta3 = minLr[1][xm + NR2 + 3] + P2;
                        CostType* Lr_p0 = Lr[0] + xd - dx*NRD2;
                        CostType* Lr_p1 = Lr[1] + xd - NRD2 + D2;
                        CostType* Lr_p2 = Lr[1] + xd + D2*2;
                        CostType* Lr_p3 = Lr[1] + xd + NRD2 + D2*3;
                        Lr_p0[-1] = Lr_p0[D] = Lr_p1[-1] = Lr_p1[D] =
                            Lr_p2[-1] = Lr_p2[D] = Lr_p3[-1] = Lr_p3[D] = MAX_COST;
                        CostType* Lr_p = Lr[0] + xd;
                        const CostType* Cp = C + x*D;
                        CostType* Sp = S + x*D;
#if CV_SSE2
                        if( useSIMD )
                        {
                            __m128i _P1 = _mm_set1_epi16((short)P1);
                            __m128i _delta0 = _mm_set1_epi16((short)delta0);
                            __m128i _delta1 = _mm_set1_epi16((short)delta1);
                            __m128i _delta2 = _mm_set1_epi16((short)delta2);
                            __m128i _delta3 = _mm_set1_epi16((short)delta3);
                            __m128i _minL0 = _mm_set1_epi16((short)MAX_COST);
                            for( d = 0; d < D; d += 8 )
                            {
                                __m128i Cpd = _mm_load_si128((const __m128i*)(Cp + d));
                                __m128i L0, L1, L2, L3;
                                L0 = _mm_load_si128((const __m128i*)(Lr_p0 + d));
                                L1 = _mm_load_si128((const __m128i*)(Lr_p1 + d));
                                L2 = _mm_load_si128((const __m128i*)(Lr_p2 + d));
                                L3 = _mm_load_si128((const __m128i*)(Lr_p3 + d));
                                L0 = _mm_min_epi16(L0, _mm_adds_epi16(_mm_loadu_si128((const __m128i*)(Lr_p0 + d - 1)), _P1));
                                L0 = _mm_min_epi16(L0, _mm_adds_epi16(_mm_loadu_si128((const __m128i*)(Lr_p0 + d + 1)), _P1));
                                L1 = _mm_min_epi16(L1, _mm_adds_epi16(_mm_loadu_si128((const __m128i*)(Lr_p1 + d - 1)), _P1));
                                L1 = _mm_min_epi16(L1, _mm_adds_epi16(_mm_loadu_si128((const __m128i*)(Lr_p1 + d + 1)), _P1));
                                L2 = _mm_min_epi16(L2, _mm_adds_epi16(_mm_loadu_si128((const __m128i*)(Lr_p2 + d - 1)), _P1));
                                L2 = _mm_min_epi16(L2, _mm_adds_epi16(_mm_loadu_si128((const __m128i*)(Lr_p2 + d + 1)), _P1));
                                L3 = _mm_min_epi16(L3, _mm_adds_epi16(_mm_loadu_si128((const __m128i*)(Lr_p3 + d - 1)), _P1));
                                L3 = _mm_min_epi16(L3, _mm_adds_epi16(_mm_loadu_si128((const __m128i*)(Lr_p3 + d + 1)), _P1));
                                L0 = _mm_min_epi16(L0, _delta0);
                                L0 = _mm_adds_epi16(_mm_subs_epi16(L0, _delta0), Cpd);
                                L1 = _mm_min_epi16(L1, _delta1);
                                L1 = _mm_adds_epi16(_mm_subs_epi16(L1, _delta1), Cpd);
                                L2 = _mm_min_epi16(L2, _delta2);
                                L2 = _mm_adds_epi16(_mm_subs_epi16(L2, _delta2), Cpd);
                                L3 = _mm_min_epi16(L3, _delta3);
                                L3 = _mm_adds_epi16(_mm_subs_epi16(L3, _delta3), Cpd);
                                _mm_store_si128( (__m128i*)(Lr_p + d), L0);
                                _mm_store_si128( (__m128i*)(Lr_p + d + D2), L1);
                                _mm_store_si128( (__m128i*)(Lr_p + d + D2*2), L2);
                                _mm_store_si128( (__m128i*)(Lr_p + d + D2*3), L3);
                                __m128i t0 = _mm_min_epi16(_mm_unpacklo_epi16(L0, L2), _mm_unpackhi_epi16(L0, L2));
                                __m128i t1 = _mm_min_epi16(_mm_unpacklo_epi16(L1, L3), _mm_unpackhi_epi16(L1, L3));
                                t0 = _mm_min_epi16(_mm_unpacklo_epi16(t0, t1), _mm_unpackhi_epi16(t0, t1));
                                _minL0 = _mm_min_epi16(_minL0, t0);
                                __m128i Sval = _mm_load_si128((const __m128i*)(Sp + d));
                                L0 = _mm_adds_epi16(L0, L1);
                                L2 = _mm_adds_epi16(L2, L3);
                                Sval = _mm_adds_epi16(Sval, L0);
                                Sval = _mm_adds_epi16(Sval, L2);
                                _mm_store_si128((__m128i*)(Sp + d), Sval);
                            }
                            _minL0 = _mm_min_epi16(_minL0, _mm_srli_si128(_minL0, 8));
                            _mm_storel_epi64((__m128i*)&minLr[0][xm], _minL0);
                        }
                        else
#endif
                        {
                            int minL0 = MAX_COST, minL1 = MAX_COST, minL2 = MAX_COST, minL3 = MAX_COST;

                            for( d = 0; d < D; d++ )
                            {
                                int Cpd = Cp[d], L0, L1, L2, L3;

                                L0 = Cpd + std::min((int)Lr_p0[d], std::min(Lr_p0[d-1] + P1, std::min(Lr_p0[d+1] + P1, delta0))) - delta0;
                                L1 = Cpd + std::min((int)Lr_p1[d], std::min(Lr_p1[d-1] + P1, std::min(Lr_p1[d+1] + P1, delta1))) - delta1;
                                L2 = Cpd + std::min((int)Lr_p2[d], std::min(Lr_p2[d-1] + P1, std::min(Lr_p2[d+1] + P1, delta2))) - delta2;
                                L3 = Cpd + std::min((int)Lr_p3[d], std::min(Lr_p3[d-1] + P1, std::min(Lr_p3[d+1] + P1, delta3))) - delta3;

                                Lr_p[d] = (CostType)L0;
                                minL0 = std::min(minL0, L0);

                                Lr_p[d + D2] = (CostType)L1;
                                minL1 = std::min(minL1, L1);

                                Lr_p[d + D2*2] = (CostType)L2;
                                minL2 = std::min(minL2, L2);

                                Lr_p[d + D2*3] = (CostType)L3;
                                minL3 = std::min(minL3, L3);

                                Sp[d] = saturate_cast<CostType>(Sp[d] + L0 + L1 + L2 + L3);
                            }
                            minLr[0][xm] = (CostType)minL0;
                            minLr[0][xm+1] = (CostType)minL1;
                            minLr[0][xm+2] = (CostType)minL2;
                            minLr[0][xm+3] = (CostType)minL3;
                        }
                    }

                    if( pass == npasses )
                    {
                        for( x = 0; x < width; x++ )
                        {
                            disp1ptr[x] = disp2ptr[x] = (DispType)INVALID_DISP_SCALED;
                            disp2cost[x] = MAX_COST;
                        }

                        for( x = width1 - 1; x >= 0; x-- )
                        {
                            CostType* Sp = S + x*D;
                            int minS = MAX_COST, bestDisp = -1;

                            if( npasses == 1 )
                            {
                                int xm = x*NR2, xd = xm*D2;

                                int minL0 = MAX_COST;
                                int delta0 = minLr[0][xm + NR2] + P2;
                                CostType* Lr_p0 = Lr[0] + xd + NRD2;
                                Lr_p0[-1] = Lr_p0[D] = MAX_COST;
                                CostType* Lr_p = Lr[0] + xd;
                                const CostType* Cp = C + x*D;
#if CV_SSE2
                                if( useSIMD )
                                {
                                    __m128i _P1 = _mm_set1_epi16((short)P1);
                                    __m128i _delta0 = _mm_set1_epi16((short)delta0);
                                    __m128i _minL0 = _mm_set1_epi16((short)minL0);
                                    __m128i _minS = _mm_set1_epi16(MAX_COST), _bestDisp = _mm_set1_epi16(-1);
                                    __m128i _d8 = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7), _8 = _mm_set1_epi16(8);
                                    for( d = 0; d < D; d += 8 )
                                    {
                                        __m128i Cpd = _mm_load_si128((const __m128i*)(Cp + d)), L0;
                                        L0 = _mm_load_si128((const __m128i*)(Lr_p0 + d));
                                        L0 = _mm_min_epi16(L0, _mm_adds_epi16(_mm_loadu_si128((const __m128i*)(Lr_p0 + d - 1)), _P1));
                                        L0 = _mm_min_epi16(L0, _mm_adds_epi16(_mm_loadu_si128((const __m128i*)(Lr_p0 + d + 1)), _P1));
                                        L0 = _mm_min_epi16(L0, _delta0);
                                        L0 = _mm_adds_epi16(_mm_subs_epi16(L0, _delta0), Cpd);
                                        _mm_store_si128((__m128i*)(Lr_p + d), L0);
                                        _minL0 = _mm_min_epi16(_minL0, L0);
                                        L0 = _mm_adds_epi16(L0, *(__m128i*)(Sp + d));
                                        _mm_store_si128((__m128i*)(Sp + d), L0);
                                        __m128i mask = _mm_cmpgt_epi16(_minS, L0);
                                        _minS = _mm_min_epi16(_minS, L0);
                                        _bestDisp = _mm_xor_si128(_bestDisp, _mm_and_si128(_mm_xor_si128(_bestDisp,_d8), mask));
                                        _d8 = _mm_adds_epi16(_d8, _8);
                                    }
                                    short CV_DECL_ALIGNED(16) bestDispBuf[8];
                                    _mm_store_si128((__m128i*)bestDispBuf, _bestDisp);
                                    _minL0 = _mm_min_epi16(_minL0, _mm_srli_si128(_minL0, 8));
                                    _minL0 = _mm_min_epi16(_minL0, _mm_srli_si128(_minL0, 4));
                                    _minL0 = _mm_min_epi16(_minL0, _mm_srli_si128(_minL0, 2));
                                    __m128i qS = _mm_min_epi16(_minS, _mm_srli_si128(_minS, 8));
                                    qS = _mm_min_epi16(qS, _mm_srli_si128(qS, 4));
                                    qS = _mm_min_epi16(qS, _mm_srli_si128(qS, 2));
                                    minLr[0][xm] = (CostType)_mm_cvtsi128_si32(_minL0);
                                    minS = (CostType)_mm_cvtsi128_si32(qS);
                                    qS = _mm_shuffle_epi32(_mm_unpacklo_epi16(qS, qS), 0);
                                    qS = _mm_cmpeq_epi16(_minS, qS);
                                    int idx = _mm_movemask_epi8(_mm_packs_epi16(qS, qS)) & 255;
                                    bestDisp = bestDispBuf[LSBTab[idx]];
                                }
                                else
#endif
                                {
                                    for( d = 0; d < D; d++ )
                                    {
                                        int L0 = Cp[d] + std::min((int)Lr_p0[d], std::min(Lr_p0[d-1] + P1, std::min(Lr_p0[d+1] + P1, delta0))) - delta0;
                                        Lr_p[d] = (CostType)L0;
                                        minL0 = std::min(minL0, L0);
                                        int Sval = Sp[d] = saturate_cast<CostType>(Sp[d] + L0);
                                        if( Sval < minS )
                                        {
                                            minS = Sval;
                                            bestDisp = d;
                                        }
                                    }
                                    minLr[0][xm] = (CostType)minL0;
                                }
                            }
                            else
                            {
                                for( d = 0; d < D; d++ )
                                {
                                    int Sval = Sp[d];
                                    if( Sval < minS )
                                    {
                                        minS = Sval;
                                        bestDisp = d;
                                    }
                                }
                            }
                            for( d = 0; d < D; d++ )
                            {
                                if( Sp[d]*(100 - uniquenessRatio) < minS*100 && std::abs(bestDisp - d) > 1 )
                                    break;
                            }
                            if( d < D )
                                continue;
                            d = bestDisp;
                            int _x2 = x + minX1 - d - minD;
                            if( disp2cost[_x2] > minS )
                            {
                                disp2cost[_x2] = (CostType)minS;
                                disp2ptr[_x2] = (DispType)(d + minD);
                            }
                            if( 0 < d && d < D-1 )
                            {
                                if(params.subpixelInterpolationMethod == CV_SIMETRICV_INTERPOLATION)
                                {
                                    double m2m1, m3m1, m3, m2, m1;
                                    m2 = Sp[d - 1];
                                    m3 = Sp[d + 1];
                                    m1 = Sp[d];
                                    m2m1 = m2 - m1;
                                    m3m1 = m3 - m1;
                                    if (!(m2m1 == 0 || m3m1 == 0))
                                    {
                                        double p;
                                        p = 0;
                                        if (m2 > m3)
                                        {
                                            p = (0.5 - 0.25 * ((m3m1 * m3m1) / (m2m1 * m2m1) + (m3m1 / m2m1)));
                                        }
                                        else
                                        {
                                            p = -1 * (0.5 - 0.25 * ((m2m1 * m2m1) / (m3m1 * m3m1) + (m2m1 / m3m1)));
                                        }
                                        if (p >= -0.5 && p <= 0.5)
                                            d = (int)(d * DISP_SCALE + p * DISP_SCALE );
                                    }
                                    else
                                    {
                                        d *= DISP_SCALE;
                                    }
                                }
                                else if(params.subpixelInterpolationMethod == CV_QUADRATIC_INTERPOLATION)
                                {
                                    // do subpixel quadratic interpolation:
                                    //   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
                                    //   then find minimum of the parabola.
                                    int denom2 = std::max(Sp[d-1] + Sp[d+1] - 2*Sp[d], 1);
                                    d = d*DISP_SCALE + ((Sp[d-1] - Sp[d+1])*DISP_SCALE + denom2)/(denom2*2);
                                }
                            }
                            else
                                d *= DISP_SCALE;
                            disp1ptr[x + minX1] = (DispType)(d + minD*DISP_SCALE);
                        }
                        for( x = minX1; x < maxX1; x++ )
                        {
                            // we round the computed disparity both towards -inf and +inf and check
                            // if either of the corresponding disparities in disp2 is consistent.
                            // This is to give the computed disparity a chance to look valid if it is.
                            int d1 = disp1ptr[x];
                            if( d1 == INVALID_DISP_SCALED )
                                continue;
                            int _d = d1 >> DISP_SHIFT;
                            int d_ = (d1 + DISP_SCALE-1) >> DISP_SHIFT;
                            int _x = x - _d, x_ = x - d_;
                            if( 0 <= _x && _x < width && disp2ptr[_x] >= minD && std::abs(disp2ptr[_x] - _d) > disp12MaxDiff &&
                                0 <= x_ && x_ < width && disp2ptr[x_] >= minD && std::abs(disp2ptr[x_] - d_) > disp12MaxDiff )
                                disp1ptr[x] = (DispType)INVALID_DISP_SCALED;
                        }
                    }
                    // now shift the cyclic buffers
                    std::swap( Lr[0], Lr[1] );
                    std::swap( minLr[0], minLr[1] );
                }
            }
        }
        class StereoBinarySGBMImpl : public StereoBinarySGBM, public Matching
        {
        public:
            StereoBinarySGBMImpl():Matching()
            {
                params = StereoBinarySGBMParams();
            }
            StereoBinarySGBMImpl( int _minDisparity, int _numDisparities, int _SADWindowSize,
                int _P1, int _P2, int _disp12MaxDiff, int _preFilterCap,
                int _uniquenessRatio, int _speckleWindowSize, int _speckleRange,
                int _mode ):Matching(_numDisparities)
            {
                params = StereoBinarySGBMParams( _minDisparity, _numDisparities, _SADWindowSize,
                    _P1, _P2, _disp12MaxDiff, _preFilterCap,
                    _uniquenessRatio, _speckleWindowSize, _speckleRange,
                    _mode );
            }
            void compute( InputArray leftarr, InputArray rightarr, OutputArray disparr )
            {
                Mat left = leftarr.getMat(), right = rightarr.getMat();
                CV_Assert( left.size() == right.size() && left.type() == right.type() &&
                    left.depth() == CV_8U );
                disparr.create( left.size(), CV_16S );
                Mat disp = disparr.getMat();
                censusImageLeft.create(left.rows,left.cols,CV_32SC4);
                censusImageRight.create(left.rows,left.cols,CV_32SC4);

                hamDist.create(left.rows, left.cols * (params.numDisparities + 1),CV_16S);

                if(params.kernelType == CV_SPARSE_CENSUS)
                {
                    censusTransform(left,right,params.kernelSize,censusImageLeft,censusImageRight,CV_SPARSE_CENSUS);
                }
                else if(params.kernelType == CV_DENSE_CENSUS)
                {
                    censusTransform(left,right,params.kernelSize,censusImageLeft,censusImageRight,CV_SPARSE_CENSUS);
                }
                else if(params.kernelType == CV_CS_CENSUS)
                {
                    symetricCensusTransform(left,right,params.kernelSize,censusImageLeft,censusImageRight,CV_CS_CENSUS);
                }
                else if(params.kernelType == CV_MODIFIED_CS_CENSUS)
                {
                    symetricCensusTransform(left,right,params.kernelSize,censusImageLeft,censusImageRight,CV_MODIFIED_CS_CENSUS);
                }
                else if(params.kernelType == CV_MODIFIED_CENSUS_TRANSFORM)
                {
                    modifiedCensusTransform(left,right,params.kernelSize,censusImageLeft,censusImageRight,CV_MODIFIED_CENSUS_TRANSFORM,0);
                }
                else if(params.kernelType == CV_MEAN_VARIATION)
                {
                    parSumsIntensityImage[0].create(left.rows, left.cols,CV_32SC4);
                    parSumsIntensityImage[1].create(left.rows, left.cols,CV_32SC4);
                    Integral[0].create(left.rows,left.cols,CV_32SC4);
                    Integral[1].create(left.rows,left.cols,CV_32SC4);
                    integral(left, parSumsIntensityImage[0],CV_32S);
                    integral(right, parSumsIntensityImage[1],CV_32S);
                    imageMeanKernelSize(parSumsIntensityImage[0], params.kernelSize,Integral[0]);
                    imageMeanKernelSize(parSumsIntensityImage[1], params.kernelSize, Integral[1]);
                    modifiedCensusTransform(left,right,params.kernelSize,censusImageLeft,censusImageRight,CV_MEAN_VARIATION,0,Integral[0], Integral[1]);
                }
                else if(params.kernelType == CV_STAR_KERNEL)
                {
                    starCensusTransform(left,right,params.kernelSize,censusImageLeft,censusImageRight);
                }

                hammingDistanceBlockMatching(censusImageLeft, censusImageRight, hamDist);

                computeDisparityBinarySGBM( left, right, disp, params, buffer,hamDist);

                if(params.regionRemoval == CV_SPECKLE_REMOVAL_AVG_ALGORITHM)
                {
                    int width = left.cols;
                    int height = left.rows;
                    if(previous_size != width * height)
                    {
                        previous_size = width * height;
                        speckleX.create(height,width,CV_32SC4);
                        speckleY.create(height,width,CV_32SC4);
                        puss.create(height,width,CV_32SC4);
                    }
                    Mat aux;
                    aux.create(height,width,CV_16S);
                    Median1x9Filter<short>(disp, aux);
                    Median9x1Filter<short>(aux,disp);
                    smallRegionRemoval<short>(disp, params.speckleWindowSize, disp);
                }
                else if(params.regionRemoval == CV_SPECKLE_REMOVAL_ALGORITHM)
                {
                    int width = left.cols;
                    int height = left.rows;
                    Mat aux;
                    aux.create(height,width,CV_16S);
                    Median1x9Filter<short>(disp, aux);
                    Median9x1Filter<short>(aux,disp);
                    if( params.speckleWindowSize > 0 )
                        filterSpeckles(disp, (params.minDisparity - 1) * StereoMatcher::DISP_SCALE, params.speckleWindowSize,
                        StereoMatcher::DISP_SCALE * params.speckleRange, buffer);
                }
            }
            int getSubPixelInterpolationMethod() const { return params.subpixelInterpolationMethod;}
            void setSubPixelInterpolationMethod(int value = CV_QUADRATIC_INTERPOLATION) { CV_Assert(value < 2); params.subpixelInterpolationMethod = value;}

            int getBinaryKernelType() const { return params.kernelType;}
            void setBinaryKernelType(int value = CV_MODIFIED_CENSUS_TRANSFORM) { CV_Assert(value < 7); params.kernelType = value; }

            int getSpekleRemovalTechnique() const { return params.regionRemoval;}
            void setSpekleRemovalTechnique(int factor = CV_SPECKLE_REMOVAL_AVG_ALGORITHM) { CV_Assert(factor < 2); params.regionRemoval = factor; }

            int getMinDisparity() const { return params.minDisparity; }
            void setMinDisparity(int minDisparity) {CV_Assert(minDisparity >= 0); params.minDisparity = minDisparity; }

            int getNumDisparities() const { return params.numDisparities; }
            void setNumDisparities(int numDisparities) { CV_Assert(numDisparities > 0); params.numDisparities = numDisparities; }

            int getBlockSize() const { return params.kernelSize; }
            void setBlockSize(int blockSize) {CV_Assert(blockSize % 2 != 0); params.kernelSize = blockSize; }

            int getSpeckleWindowSize() const { return params.speckleWindowSize; }
            void setSpeckleWindowSize(int speckleWindowSize) {CV_Assert(speckleWindowSize >= 0); params.speckleWindowSize = speckleWindowSize; }

            int getSpeckleRange() const { return params.speckleRange; }
            void setSpeckleRange(int speckleRange) { CV_Assert(speckleRange >= 0); params.speckleRange = speckleRange; }

            int getDisp12MaxDiff() const { return params.disp12MaxDiff; }
            void setDisp12MaxDiff(int disp12MaxDiff) {CV_Assert(disp12MaxDiff > 0); params.disp12MaxDiff = disp12MaxDiff; }

            int getPreFilterCap() const { return params.preFilterCap; }
            void setPreFilterCap(int preFilterCap) { CV_Assert(preFilterCap > 0); params.preFilterCap = preFilterCap; }

            int getUniquenessRatio() const { return params.uniquenessRatio; }
            void setUniquenessRatio(int uniquenessRatio) { CV_Assert(uniquenessRatio >= 0); params.uniquenessRatio = uniquenessRatio; }

            int getP1() const { return params.P1; }
            void setP1(int P1) { CV_Assert(P1 > 0); params.P1 = P1; }

            int getP2() const { return params.P2; }
            void setP2(int P2) {CV_Assert(P2 > 0); CV_Assert(P2 >= 2 * params.P1); params.P2 = P2; }

            int getMode() const { return params.mode; }
            void setMode(int mode) { params.mode = mode; }

            void write(FileStorage& fs) const
            {
                fs << "name" << name_
                    << "minDisparity" << params.minDisparity
                    << "numDisparities" << params.numDisparities
                    << "blockSize" << params.kernelSize
                    << "speckleWindowSize" << params.speckleWindowSize
                    << "speckleRange" << params.speckleRange
                    << "disp12MaxDiff" << params.disp12MaxDiff
                    << "preFilterCap" << params.preFilterCap
                    << "uniquenessRatio" << params.uniquenessRatio
                    << "P1" << params.P1
                    << "P2" << params.P2
                    << "mode" << params.mode;
            }

            void read(const FileNode& fn)
            {
                FileNode n = fn["name"];
                CV_Assert( n.isString() && String(n) == name_ );
                params.minDisparity = (int)fn["minDisparity"];
                params.numDisparities = (int)fn["numDisparities"];
                params.kernelSize = (int)fn["blockSize"];
                params.speckleWindowSize = (int)fn["speckleWindowSize"];
                params.speckleRange = (int)fn["speckleRange"];
                params.disp12MaxDiff = (int)fn["disp12MaxDiff"];
                params.preFilterCap = (int)fn["preFilterCap"];
                params.uniquenessRatio = (int)fn["uniquenessRatio"];
                params.P1 = (int)fn["P1"];
                params.P2 = (int)fn["P2"];
                params.mode = (int)fn["mode"];
            }

            StereoBinarySGBMParams params;
            Mat buffer;
            static const char* name_;
            Mat censusImageLeft;
            Mat censusImageRight;
            Mat partialSumsLR;
            Mat agregatedHammingLRCost;
            Mat hamDist;
            Mat parSumsIntensityImage[2];
            Mat Integral[2];
        };

        const char* StereoBinarySGBMImpl::name_ = "StereoBinaryMatcher.SGBM";

        Ptr<StereoBinarySGBM> StereoBinarySGBM::create(int minDisparity, int numDisparities, int kernelSize,
            int P1, int P2, int disp12MaxDiff,
            int preFilterCap, int uniquenessRatio,
            int speckleWindowSize, int speckleRange,
            int mode)
        {
            return Ptr<StereoBinarySGBM>(
                new StereoBinarySGBMImpl(minDisparity, numDisparities, kernelSize,
                P1, P2, disp12MaxDiff,
                preFilterCap, uniquenessRatio,
                speckleWindowSize, speckleRange,
                mode));
        }

        typedef cv::Point_<short> Point2s;
    }
}
