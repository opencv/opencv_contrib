//By downloading, copying, installing or using the software you agree to this license.
//If you do not agree to this license, do not download, install,
//copy or use the software.
//
//
//                          License Agreement
//               For Open Source Computer Vision Library
//                       (3-clause BSD License)
//
//Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
//Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
//Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
//Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
//Copyright (C) 2015, OpenCV Foundation, all rights reserved.
//Copyright (C) 2015, Itseez Inc., all rights reserved.
//Third party copyrights are property of their respective owners.
//
//Redistribution and use in source and binary forms, with or without modification,
//are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * Neither the names of the copyright holders nor the names of the contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
//This software is provided by the copyright holders and contributors "as is" and
//any express or implied warranties, including, but not limited to, the implied
//warranties of merchantability and fitness for a particular purpose are disclaimed.
//In no event shall copyright holders or contributors be liable for any direct,
//indirect, incidental, special, exemplary, or consequential damages
//(including, but not limited to, procurement of substitute goods or services;
//loss of use, data, or profits; or business interruption) however caused
//and on any theory of liability, whether in contract, strict liability,
//or tort (including negligence or otherwise) arising in any way out of
//the use of this software, even if advised of the possibility of such damage.

/*****************************************************************************************************************\
*   The interface contains the main methods for computing the matching between the left and right images	      *
*                                                                                                                 *
\******************************************************************************************************************/
#include "precomp.hpp"
#include <stdint.h>

#ifndef _OPENCV_MATCHING_HPP_
#define _OPENCV_MATCHING_HPP_
#ifdef __cplusplus

namespace cv
{
    namespace stereo
    {
        class Matching
        {
        private:
            //arrays used in the region removal
            int specklePointX[1000001];
            int specklePointY[1000001];
            long long pus[2000001];
            //!The maximum disparity
            int maxDisparity;
            //!the factor by which we are multiplying the disparity
            int scallingFactor;
            //!the confidence to which a min disparity found is good or not
            double confidenceCheck;
            //!the LUT used in case SSE is not available
            int hamLut[65537];
            //!function for refining the disparity at sub pixel using simetric v
            static double symetricVInterpolation(int *c, int iwjp, int widthDisp, int winDisp,const int search_region);
            //!function used for getting the minimum disparity from the cost volume"
            static int minim(int *c, int iwpj, int widthDisp,const double confidenceCheck, const int search_region);
            //!a pre processing function that generates the Hamming LUT in case the algorithm will ever be used on platform where SSE is not available
            void hammingLut();
            //!the class used in computing the hamming distance
            class hammingDistance : public ParallelLoopBody
            {
            private:
                int *left, *right, *c;
                int v,kernelSize, width, height,_stride;
                int MASK;
                int *hammLut;
            public :
                hammingDistance(const Mat &leftImage, const Mat &rightImage, int *cost, int maxDisp, int kerSize, int *hammingLUT):
                    left((int *)leftImage.data), right((int *)rightImage.data), c(cost), v(maxDisp),kernelSize(kerSize),width(leftImage.cols), height(leftImage.rows), _stride((int)leftImage.step1()), MASK(65535), hammLut(hammingLUT){}
                void operator()(const cv::Range &r) const {
                    for (int i = r.start; i <= r.end ; i++)
                    {
                        int iw = i * width;
                        for (int j = kernelSize; j < width - kernelSize; j++)
                        {
                            int j2;
                            int xorul;
                            int iwj;
                            iwj = iw + j;
                            for (int d = 0; d <= v; d++)
                            {
                                j2 = (0 > j - d) ? (0) : (j - d);
                                xorul = left[(iwj)] ^ right[(iw + j2)];
#if CV_SSE4_1
                                c[(iwj)* (v + 1) + d] = _mm_popcnt_u32(xorul);
#else
                                c[(iwj)* (v + 1) + d] = hammLut[xorul & MASK] + hammLut[xorul >> 16];
#endif
                            }
                        }
                    }
                }
            };
            //!preprocessing used for agregation
            class costGatheringHorizontal:public ParallelLoopBody
            {
            private:
                int *c, *ham;
                int width, maxDisp;
            public:
                costGatheringHorizontal(const Mat &hamimg,const int maxDispa, Mat &output)
                {
                    ham = (int *)hamimg.data;
                    c = (int *)output.data;
                    maxDisp = maxDispa;
                    width = output.cols / ( maxDisp + 1) - 1;
                }
                void operator()(const cv::Range &r) const {
                    for (int i = r.start; i <= r.end; i++)
                    {
                        int iw = i * width;
                        int iwi = (i - 1) * width;
                        for (int j = 1; j <= width; j++)
                        {
                            int iwj = (iw + j) * (maxDisp + 1);
                            int iwjmu = (iw + j - 1) * (maxDisp + 1);
                            int iwijmu = (iwi + j - 1) * (maxDisp + 1);
                            for (int d = 0; d <= maxDisp; d++)
                            {
                                c[iwj + d] = ham[iwijmu + d] + c[iwjmu + d];
                            }
                        }
                    }
                }
            };
            //!cost aggregation
            class agregateCost:public ParallelLoopBody
            {
            private:
                int win;
                int *c, *parSum;
                int maxDisp,width, height;
            public:
                agregateCost(const Mat &partialSums, int windowSize, int maxDispa, Mat &cost)
                {
                    win = windowSize / 2;
                    c = (int *)cost.data;
                    maxDisp = maxDispa;
                    width = cost.cols / ( maxDisp + 1) - 1;
                    height = cost.rows - 1;
                    parSum = (int *)partialSums.data;
                }
                void operator()(const cv::Range &r) const {
                    for (int i = r.start; i <= r.end; i++)
                    {
                        int iwi = (i - 1) * width;
                        for (int j = win + 1; j <= width - win - 1; j++)
                        {
                            int w1 = ((i + win + 1) * width + j + win) * (maxDisp + 1);
                            int w2 = ((i - win) * width + j - win - 1) * (maxDisp + 1);
                            int w3 = ((i + win + 1) * width + j - win - 1) * (maxDisp + 1);
                            int w4 = ((i - win) * width + j + win) * (maxDisp + 1);
                            int w = (iwi + j - 1) * (maxDisp + 1);
                            for (int d = 0; d <= maxDisp; d++)
                            {
                                c[w + d] = parSum[w1 + d] + parSum[w2 + d]
                                - parSum[w3 + d] - parSum[w4 + d];
                            }
                        }
                    }
                }
            };
            //!class that is responsable for generating the disparity map
            class makeMap:public ParallelLoopBody
            {
            private:
                //enum used to notify wether we are searching on the vertical ie (lr) or diagonal (rl)
                enum {CV_VERTICAL_SEARCH, CV_DIAGONAL_SEARCH};
                int width,disparity,scallingFact,th;
                double confCheck;
                uint8_t *map;
                int *c;
            public:
                makeMap(const Mat &costVolume, int threshold, int maxDisp, double confidence,int scale, Mat &mapFinal)
                {
                    c = (int *)costVolume.data;
                    map = mapFinal.data;
                    disparity = maxDisp;
                    width = costVolume.cols / ( disparity + 1) - 1;
                    th = threshold;
                    scallingFact = scale;
                    confCheck = confidence;
                }
                void operator()(const cv::Range &r) const {
                    for (int i = r.start; i <= r.end ; i++)
                    {
                        int lr;
                        int v = -1;
                        double p1, p2;
                        int iw = i * width;
                        for (int j = 0; j < width; j++)
                        {
                            lr = Matching:: minim(c, iw + j, disparity + 1, confCheck,CV_VERTICAL_SEARCH);
                            if (lr != -1)
                            {
                                v = Matching::minim(c, iw + j - lr, disparity + 1, confCheck,CV_DIAGONAL_SEARCH);
                                if (v != -1)
                                {
                                    p1 = Matching::symetricVInterpolation(c, iw + j - lr, disparity + 1, v,CV_DIAGONAL_SEARCH);
                                    p2 = Matching::symetricVInterpolation(c, iw + j, disparity + 1, lr,CV_VERTICAL_SEARCH);
                                    if (abs(p1 - p2) <= th)
                                        map[iw + j] = (uint8_t)((p2)* scallingFact);
                                    else
                                    {
                                        map[iw + j] = 0;
                                    }
                                }
                                else
                                {
                                    if (width - j <= disparity)
                                    {
                                        p2 = Matching::symetricVInterpolation(c, iw + j, disparity + 1, lr,CV_VERTICAL_SEARCH);
                                        map[iw + j] = (uint8_t)(p2* scallingFact);
                                    }
                                }
                            }
                            else
                            {
                                map[iw + j] = 0;
                            }
                        }
                    }
                }
            };
            //!median 1x9 paralelized filter
            class Median1x9:public ParallelLoopBody
            {
            private:
                uint8_t *original;
                uint8_t *filtered;
                int height, width,_stride;
            public:
                Median1x9(const Mat &originalImage, Mat &filteredImage)
                {
                    original = originalImage.data;
                    filtered = filteredImage.data;
                    height = originalImage.rows;
                    width = originalImage.cols;
                    _stride = (int)originalImage.step;
                }
                void operator()(const cv::Range &r) const{
                    for (int m = r.start; m <= r.end; m++)
                    {
                        for (int n = 4; n < width - 4; ++n)
                        {
                            int k = 0;
                            uint8_t window[9];
                            for (int i = n - 4; i <= n + 4; ++i)
                                window[k++] = original[m * _stride + i];
                            for (int j = 0; j < 5; ++j)
                            {
                                int min = j;
                                for (int l = j + 1; l < 9; ++l)
                                    if (window[l] < window[min])
                                        min = l;
                                const uint8_t temp = window[j];
                                window[j] = window[min];
                                window[min] = temp;
                            }
                            filtered[m  * _stride + n] = window[4];
                        }
                    }
                }
            };
            //!median 9x1 paralelized filter
            class Median9x1:public ParallelLoopBody
            {
            private:
                uint8_t *original;
                uint8_t *filtered;
                int height, width, _stride;
            public:
                Median9x1(const Mat &originalImage, Mat &filteredImage)
                {
                    original = originalImage.data;
                    filtered = filteredImage.data;
                    height = originalImage.rows;
                    width = originalImage.cols;
                    _stride = (int)originalImage.step;
                }
                void operator()(const Range &r) const{
                    for (int n = r.start; n <= r.end; ++n)
                    {
                        for (int m = 4; m < height - 4; ++m)
                        {
                            int k = 0;
                            uint8_t window[9];
                            for (int i = m - 4; i <= m + 4; ++i)
                                window[k++] = original[i * _stride + n];
                            for (int j = 0; j < 5; j++)
                            {
                                int min = j;
                                for (int l = j + 1; l < 9; ++l)
                                    if (window[l] < window[min])
                                        min = l;
                                const uint8_t temp = window[j];
                                window[j] = window[min];
                                window[min] = temp;
                            }
                            filtered[m  * _stride + n] = window[4];
                        }
                    }
                }
            };
        protected:
            //!method for setting the maximum disparity
            void setMaxDisparity(int val);
            //!method for getting the disparity
            int getMaxDisparity();
            //!method for setting the scalling factor
            void setScallingFactor(int val);
            //!method for getting the scalling factor
            int getScallingFactor();
            //!setter for the confidence check
            void setConfidence(double val);
            //!getter for confidence check
            double getConfidence();
            //!method for computing the hamming difference
            void hammingDistanceBlockMatching(const Mat &left, const Mat &right, Mat &c, const int kernelSize = 9);
            //!precomputation done on the cost volume to efficiently compute the block matching
            void costGathering(const Mat &hammingDistanceCost, Mat &c);
            //the aggregation on the cost volume
            void blockAgregation(const Mat &parSum, int windowSize, Mat &c);
            //!function for generating disparity maps at sub pixel level
            /* costVolume - represents the cost volume
            * width, height - represent the width and height of the iage
            *disparity - represents the maximum disparity
            *map - is the disparity map that will result
            *th - is the LR threshold
            */
            void dispartyMapFormation(const Mat &costVolume, Mat &map, int th);
            void smallRegionRemoval(const Mat &input, int t, Mat &out);
        public:
            static void Median1x9Filter(const Mat &inputImage, Mat &outputImage);
            static void Median9x1Filter(const Mat &inputImage, Mat &outputImage);
            //!constructor for the matching class
            //!maxDisp - represents the maximum disparity
            //!a median filter that has proven to work a bit better especially when applied on disparity maps
            Matching(int maxDisp, int scallingFactor = 4,int confidenceCheck = 6);
            Matching(void);
            ~Matching(void);
        };
    }
}
#endif
#endif
/*End of file*/