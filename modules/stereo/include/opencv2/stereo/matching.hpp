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
            //!The maximum disparity
            int maxDisparity;
            //!the factor by which we are multiplying the disparity
            int scallingFactor;
            //!the confidence to which a min disparity found is good or not
            double confidenceCheck;
            //!the LUT used in case SSE is not available
            int hamLut[65537];
            //!function used for getting the minimum disparity from the cost volume"
            static int minim(short *c, int iwpj, int widthDisp,const double confidence, const int search_region)
            {
                double mini, mini2, mini3;
                mini = mini2 = mini3 = DBL_MAX;
                int index = 0;
                int iw = iwpj;
                int widthDisp2;
                widthDisp2 = widthDisp;
                widthDisp -= 1;
                for (int i = 0; i <= widthDisp; i++)
                {
                    if (c[(iw + i * search_region) * widthDisp2 + i] < mini)
                    {
                        mini3 = mini2;
                        mini2 = mini;
                        mini = c[(iw + i * search_region) * widthDisp2 + i];
                        index = i;
                    }
                    else if (c[(iw + i * search_region) * widthDisp2 + i] < mini2)
                    {
                        mini3 = mini2;
                        mini2 = c[(iw + i * search_region) * widthDisp2 + i];
                    }
                    else if (c[(iw + i * search_region) * widthDisp2 + i] < mini3)
                    {
                        mini3 = c[(iw + i * search_region) * widthDisp2 + i];
                    }
                }
                if(mini != 0)
                {
                    if (mini3 / mini <= confidence)
                        return index;
                }
                return -1;
            }
            //!Interpolate in order to obtain better results
            //!function for refining the disparity at sub pixel using simetric v
            static double symetricVInterpolation(short *c, int iwjp, int widthDisp, int winDisp,const int search_region)
            {
                if (winDisp == 0 || winDisp == widthDisp - 1)
                    return winDisp;
                double m2m1, m3m1, m3, m2, m1;
                m2 = c[(iwjp + (winDisp - 1) * search_region) * widthDisp + winDisp - 1];
                m3 = c[(iwjp + (winDisp + 1) * search_region)* widthDisp + winDisp + 1];
                m1 = c[(iwjp + winDisp * search_region) * widthDisp + winDisp];
                m2m1 = m2 - m1;
                m3m1 = m3 - m1;
                if (m2m1 == 0 || m3m1 == 0) return winDisp;
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
                    p = winDisp + p;
                return p;
            }
            //!a pre processing function that generates the Hamming LUT in case the algorithm will ever be used on platform where SSE is not available
            void hammingLut()
            {
                for (int i = 0; i <= 65536; i++)
                {
                    int dist = 0;
                    int j = i;
                    //we number the bits from our number
                    while (j)
                    {
                        dist = dist + 1;
                        j = j & (j - 1);
                    }
                    hamLut[i] = dist;
                }
            }
            //!the class used in computing the hamming distance
            class hammingDistance : public ParallelLoopBody
            {
            private:
                int *left, *right;
                short *c;
                int v,kernelSize, width;
                int MASK;
                int *hammLut;
            public :
                hammingDistance(const Mat &leftImage, const Mat &rightImage, short *cost, int maxDisp, int kerSize, int *hammingLUT):
                    left((int *)leftImage.data), right((int *)rightImage.data), c(cost), v(maxDisp),kernelSize(kerSize),width(leftImage.cols), MASK(65535), hammLut(hammingLUT){}
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
#if CV_POPCNT
                                if (checkHardwareSupport(CV_CPU_POPCNT))
                                {
                                    c[(iwj)* (v + 1) + d] = (short)_mm_popcnt_u32(xorul);
                                }
                                else
#endif
                                {
                                    c[(iwj)* (v + 1) + d] = (short)(hammLut[xorul & MASK] + hammLut[(xorul >> 16) & MASK]);
                                }
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
                short *c, *parSum;
                int maxDisp,width, height;
            public:
                agregateCost(const Mat &partialSums, int windowSize, int maxDispa, Mat &cost)
                {
                    win = windowSize / 2;
                    c = (short *)cost.data;
                    maxDisp = maxDispa;
                    width = cost.cols / ( maxDisp + 1) - 1;
                    height = cost.rows - 1;
                    parSum = (short *)partialSums.data;
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
                short *c;
            public:
                makeMap(const Mat &costVolume, int threshold, int maxDisp, double confidence,int scale, Mat &mapFinal)
                {
                    c = (short *)costVolume.data;
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
            template <typename T>
            class Median1x9:public ParallelLoopBody
            {
            private:
                T *original;
                T *filtered;
                int height, width;
            public:
                Median1x9(const Mat &originalImage, Mat &filteredImage)
                {
                    original = (T *)originalImage.data;
                    filtered = (T *)filteredImage.data;
                    height = originalImage.rows;
                    width = originalImage.cols;
                }
                void operator()(const cv::Range &r) const{
                    for (int m = r.start; m <= r.end; m++)
                    {
                        for (int n = 4; n < width - 4; ++n)
                        {
                            int k = 0;
                            T window[9];
                            for (int i = n - 4; i <= n + 4; ++i)
                                window[k++] = original[m * width + i];
                            for (int j = 0; j < 5; ++j)
                            {
                                int min = j;
                                for (int l = j + 1; l < 9; ++l)
                                    if (window[l] < window[min])
                                        min = l;
                                const T temp = window[j];
                                window[j] = window[min];
                                window[min] = temp;
                            }
                            filtered[m  * width + n] = window[4];
                        }
                    }
                }
            };
            //!median 9x1 paralelized filter
            template <typename T>
            class Median9x1:public ParallelLoopBody
            {
            private:
                T *original;
                T *filtered;
                int height, width;
            public:
                Median9x1(const Mat &originalImage, Mat &filteredImage)
                {
                    original = (T *)originalImage.data;
                    filtered = (T *)filteredImage.data;
                    height = originalImage.rows;
                    width = originalImage.cols;
                }
                void operator()(const Range &r) const{
                    for (int n = r.start; n <= r.end; ++n)
                    {
                        for (int m = 4; m < height - 4; ++m)
                        {
                            int k = 0;
                            T window[9];
                            for (int i = m - 4; i <= m + 4; ++i)
                                window[k++] = original[i * width + n];
                            for (int j = 0; j < 5; j++)
                            {
                                int min = j;
                                for (int l = j + 1; l < 9; ++l)
                                    if (window[l] < window[min])
                                        min = l;
                                const T temp = window[j];
                                window[j] = window[min];
                                window[min] = temp;
                            }
                            filtered[m  * width + n] = window[4];
                        }
                    }
                }
            };
        protected:
            //arrays used in the region removal
            Mat speckleY;
            Mat speckleX;
            Mat puss;
            //int *specklePointX;
            //int *specklePointY;
            //long long *pus;
            int previous_size;
            //!method for setting the maximum disparity
            void setMaxDisparity(int val)
            {
                CV_Assert(val > 10);
                this->maxDisparity = val;
            }
            //!method for getting the disparity
            int getMaxDisparity()
            {
                return this->maxDisparity;
            }
            //! a number by which the disparity will be multiplied for better display
            void setScallingFactor(int val)
            {
                CV_Assert(val > 0);
                this->scallingFactor = val;
            }
            //!method for getting the scalling factor
            int getScallingFactor()
            {
                return scallingFactor;
            }
            //!setter for the confidence check
            void setConfidence(double val)
            {
                CV_Assert(val >= 1);
                this->confidenceCheck = val;
            }
            //getter for confidence check
            double getConfidence()
            {
                return confidenceCheck;
            }
            //! Hamming distance computation method
            //! leftImage and rightImage are the two transformed images
            //! the cost is the resulted cost volume and kernel Size is the size of the matching window
            void hammingDistanceBlockMatching(const Mat &leftImage, const Mat &rightImage, Mat &cost, const int kernelSize= 9)
            {
                CV_Assert(leftImage.cols == rightImage.cols);
                CV_Assert(leftImage.rows == rightImage.rows);
                CV_Assert(kernelSize % 2 != 0);
                CV_Assert(cost.rows == leftImage.rows);
                CV_Assert(cost.cols / (maxDisparity + 1) == leftImage.cols);
                short *c = (short *)cost.data;
                memset(c, 0, sizeof(c[0]) * leftImage.cols * leftImage.rows * (maxDisparity + 1));
                parallel_for_(cv::Range(kernelSize / 2,leftImage.rows - kernelSize / 2), hammingDistance(leftImage,rightImage,(short *)cost.data,maxDisparity,kernelSize / 2,hamLut));
            }
            //preprocessing the cost volume in order to get it ready for aggregation
            void costGathering(const Mat &hammingDistanceCost, Mat &cost)
            {
                CV_Assert(hammingDistanceCost.rows == hammingDistanceCost.rows);
                CV_Assert(hammingDistanceCost.type() == CV_16S);
                CV_Assert(cost.type() == CV_16S);
                int maxDisp = maxDisparity;
                int width = cost.cols / ( maxDisp + 1) - 1;
                int height = cost.rows - 1;
                short *c = (short *)cost.data;
                short *ham = (short *)hammingDistanceCost.data;
                memset(c, 0, sizeof(c[0]) * (width + 1) * (height + 1) * (maxDisp + 1));
                for (int i = 1; i <= height; i++)
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
                for (int i = 1; i <= height; i++)
                {
                    for (int j = 1; j <= width; j++)
                    {
                        int iwj = (i * width + j) * (maxDisp + 1);
                        int iwjmu = ((i - 1)  * width + j) * (maxDisp + 1);
                        for (int d = 0; d <= maxDisp; d++)
                        {
                            c[iwj + d] += c[iwjmu + d];
                        }
                    }
                }
            }
            //!The aggregation on the cost volume
            void blockAgregation(const Mat &partialSums, int windowSize, Mat &cost)
            {
                CV_Assert(windowSize % 2 != 0);
                CV_Assert(partialSums.rows == cost.rows);
                CV_Assert(partialSums.cols == cost.cols);
                int win = windowSize / 2;
                short *c = (short *)cost.data;
                int maxDisp = maxDisparity;
                int width = cost.cols / ( maxDisp + 1) - 1;
                int height = cost.rows - 1;
                memset(c, 0, sizeof(c[0]) * width * height * (maxDisp + 1));
                parallel_for_(cv::Range(win + 1,height - win - 1), agregateCost(partialSums,windowSize,maxDisp,cost));
            }
            //!remove small regions that have an area smaller than t, we fill the region with the average of the good pixels around it
            template <typename T>
            void smallRegionRemoval(const Mat &currentMap, int t, Mat &out)
            {
                CV_Assert(currentMap.cols == out.cols);
                CV_Assert(currentMap.rows == out.rows);
                CV_Assert(t >= 0);
                int *pus = (int *)puss.data;
                int *specklePointX = (int *)speckleX.data;
                int *specklePointY = (int *)speckleY.data;
                memset(pus, 0, previous_size * sizeof(pus[0]));
                T *map = (T *)currentMap.data;
                T *outputMap = (T *)out.data;
                int height = currentMap.rows;
                int width = currentMap.cols;
                T k = 1;
                int st, dr;
                int di[] = { -1, -1, -1, 0, 1, 1, 1, 0 },
                    dj[] = { -1, 0, 1, 1, 1, 0, -1, -1 };
                int speckle_size = 0;
                st = 0;
                dr = 0;
                for (int i = 1; i < height - 1; i++)
                {
                    int iw = i * width;
                    for (int j = 1; j < width - 1; j++)
                    {
                        if (map[iw + j] != 0)
                        {
                            outputMap[iw + j] = map[iw + j];
                        }
                        else if (map[iw + j] == 0)
                        {
                            T nr = 1;
                            T avg = 0;
                            speckle_size = dr;
                            specklePointX[dr] = i;
                            specklePointY[dr] = j;
                            pus[i * width + j] = 1;
                            dr++;
                            map[iw + j] = k;
                            while (st < dr)
                            {
                                int ii = specklePointX[st];
                                int jj = specklePointY[st];
                                //going on 8 directions
                                for (int d = 0; d < 8; d++)
                                {//if insisde
                                    if (ii + di[d] >= 0 && ii + di[d] < height && jj + dj[d] >= 0 && jj + dj[d] < width &&
                                        pus[(ii + di[d]) * width + jj + dj[d]] == 0)
                                    {
                                        T val = map[(ii + di[d]) * width + jj + dj[d]];
                                        if (val == 0)
                                        {
                                            map[(ii + di[d]) * width + jj + dj[d]] = k;
                                            specklePointX[dr] = (ii + di[d]);
                                            specklePointY[dr] = (jj + dj[d]);
                                            dr++;
                                            pus[(ii + di[d]) * width + jj + dj[d]] = 1;
                                        }//this means that my point is a good point to be used in computing the final filling value
                                        else if (val >= 1 && val < 250)
                                        {
                                            avg += val;
                                            nr++;
                                        }
                                    }
                                }
                                st++;
                            }//if hole size is smaller than a specified threshold we fill the respective hole with the average of the good neighbours
                            if (st - speckle_size <= t)
                            {
                                T fillValue = (T)(avg / nr);
                                while (speckle_size < st)
                                {
                                    int ii = specklePointX[speckle_size];
                                    int jj = specklePointY[speckle_size];
                                    outputMap[ii * width + jj] = fillValue;
                                    speckle_size++;
                                }
                            }
                        }
                    }
                }
            }
            //!Method responsible for generating the disparity map
            //!function for generating disparity maps at sub pixel level
            /* costVolume - represents the cost volume
            * width, height - represent the width and height of the iage
            *disparity - represents the maximum disparity
            *map - is the disparity map that will result
            *th - is the LR threshold
            */
            void dispartyMapFormation(const Mat &costVolume, Mat &mapFinal, int th)
            {
                uint8_t *map = mapFinal.data;
                int disparity = maxDisparity;
                int width = costVolume.cols / ( disparity + 1) - 1;
                int height = costVolume.rows - 1;
                memset(map, 0, sizeof(map[0]) * width * height);
                parallel_for_(Range(0,height - 1), makeMap(costVolume,th,disparity,confidenceCheck,scallingFactor,mapFinal));
            }
        public:
            //!a median filter of 1x9 and 9x1
            //!1x9 median filter
            template<typename T>
            void Median1x9Filter(const Mat &originalImage, Mat &filteredImage)
            {
                CV_Assert(originalImage.rows == filteredImage.rows);
                CV_Assert(originalImage.cols == filteredImage.cols);
                parallel_for_(Range(1,originalImage.rows - 2), Median1x9<T>(originalImage,filteredImage));
            }
            //!9x1 median filter
            template<typename T>
            void Median9x1Filter(const Mat &originalImage, Mat &filteredImage)
            {
                CV_Assert(originalImage.cols == filteredImage.cols);
                CV_Assert(originalImage.cols == filteredImage.cols);
                parallel_for_(Range(1,originalImage.cols - 2), Median9x1<T>(originalImage,filteredImage));
            }
            //!constructor for the matching class
            //!maxDisp - represents the maximum disparity
            Matching(void)
            {
                hammingLut();
            }
            ~Matching(void)
            {
            }
            //constructor for the matching class
            //maxDisp - represents the maximum disparity
            //confidence - represents the confidence check
            Matching(int maxDisp, int scalling = 4, int confidence = 6)
            {
                //set the maximum disparity
                setMaxDisparity(maxDisp);
                //set scalling factor
                setScallingFactor(scalling);
                //set the value for the confidence
                setConfidence(confidence);
                //generate the hamming lut in case SSE is not available
                hammingLut();
            }
        };
    }
}
#endif
#endif
/*End of file*/
