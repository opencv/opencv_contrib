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
#include "precomp.hpp"
#include "matching.hpp"

namespace cv
{
    namespace stereo
    {
        //null constructor
        Matching::Matching(void)
        {
        }
        Matching::~Matching(void)
        {
        }
        //constructor for the matching class
        //maxDisp - represents the maximum disparity
        Matching::Matching(int maxDisp, int scalling, int confidence)
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
        //!method for setting the maximum disparity
        void Matching::setMaxDisparity(int val)
        {
            CV_Assert(val > 10);
            this->maxDisparity = val;
        }
        //!method for getting the disparity
        int Matching::getMaxDisparity()
        {
            return this->maxDisparity;
        }
        void Matching::setScallingFactor(int val)
        {
            CV_Assert(val > 0);
            this->scallingFactor = val;
        }
        //!method for getting the scalling factor
        int Matching::getScallingFactor()
        {
            return scallingFactor;
        }
        //! Hamming distance computation method
        //! leftImage and rightImage are the two transformed images
        //! the cost is the resulted cost volume and kernel Size is the size of the matching window
        void Matching::hammingDistanceBlockMatching(const Mat &leftImage, const Mat &rightImage, Mat &cost, const int kernelSize)
        {
            CV_Assert(leftImage.cols == rightImage.cols);
            CV_Assert(leftImage.rows == rightImage.rows);
            CV_Assert(kernelSize % 2 != 0);
            CV_Assert(cost.rows == leftImage.rows);
            CV_Assert(cost.cols / (maxDisparity + 1) == leftImage.cols);
            cost.setTo(0);
            //int *c = (int *)cost.data;
            //memset(c, 0, sizeof(c[0]) * leftImage.cols * leftImage.rows * (maxDisparity + 1));
            parallel_for_(cv::Range(kernelSize / 2,leftImage.rows - kernelSize / 2), hammingDistance(leftImage,rightImage,(int *)cost.data,maxDisparity,kernelSize / 2,hamLut));
        }
        //preprocessing the cost volume in order to get it ready for aggregation
        void Matching::costGathering(const Mat &hammingDistanceCost, Mat &cost)
        {
            CV_Assert(hammingDistanceCost.rows == hammingDistanceCost.rows);
            CV_Assert(hammingDistanceCost.type() == CV_32SC4);
            CV_Assert(cost.type() == CV_32SC4);
            cost.setTo(0);
            int maxDisp = maxDisparity;
            int width = cost.cols / ( maxDisp + 1) - 1;
            int height = cost.rows - 1;
            int *c = (int *)cost.data;
            //memset(c, 0, sizeof(c[0]) * (width + 1) * (height + 1) * (maxDisp + 1));
            parallel_for_(cv::Range(1,height), costGatheringHorizontal(hammingDistanceCost,maxDisparity,cost));
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
        void Matching::blockAgregation(const Mat &partialSums, int windowSize, Mat &cost)
        {
            CV_Assert(windowSize % 2 != 0);
            CV_Assert(partialSums.rows == cost.rows);
            CV_Assert(partialSums.cols == cost.cols);
            cost.setTo(0);
            int win = windowSize / 2;
            //int *c = (int *)cost.data;
            int maxDisp = maxDisparity;
            //int width = cost.cols / ( maxDisp + 1) - 1;
            int height = cost.rows - 1;
            //memset(c, 0, sizeof(c[0]) * width * height * (maxDisp + 1));
            parallel_for_(cv::Range(win + 1,height - win - 1), agregateCost(partialSums,windowSize,maxDisp,cost));
        }
        //!Finding the correct disparity from the cost volume, we also make a confidence check
        int Matching ::minim(int *c, int iwpj, int widthDisp,const double confidenceCheck, const int search_region)
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
            if (mini3 / mini <= confidenceCheck)
                return index;
            return -1;
        }
        //!Interpolate in order to obtain better results
        double Matching::symetricVInterpolation(int *c, int iwjp, int widthDisp, int winDisp,const int search_region)
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
        //!Generate the hamming LUT
        void Matching::hammingLut()
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
        //!remove small regions that have an area smaller than t, we fill the region with the average of the good pixels around it
        void Matching::smallRegionRemoval(const Mat &currentMap, int t, Mat &out)
        {
            CV_Assert(currentMap.cols == out.cols);
            CV_Assert(currentMap.rows == out.rows);
            CV_Assert(t > 0);
            memset(pus, 0, 2000000 * sizeof(pus[0]));
            uint8_t *map = currentMap.data;
            uint8_t *outputMap = out.data;
            int height = currentMap.rows;
            int width = currentMap.cols;
            uint8_t k = 1;
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
                        int nr = 1;
                        int avg = 0;
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
                                    int val = map[(ii + di[d]) * width + jj + dj[d]];
                                    if (val == 0)
                                    {
                                        map[(ii + di[d]) * width + jj + dj[d]] = k;
                                        specklePointX[dr] = (ii + di[d]);
                                        specklePointY[dr] = (jj + dj[d]);
                                        dr++;
                                        pus[(ii + di[d]) * width + jj + dj[d]] = 1;
                                    }//this means that my point is a good point to be used in computing the final filling value
                                    else if (val > 2 && val < 250)
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
                            uint8_t fillValue = (uint8_t)(avg / nr);
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
        //!setting the confidence to a certain value
        void Matching ::setConfidence(double val)
        {
            CV_Assert(val >= 1);
            this->confidenceCheck = val;
        }
        //getter for confidence check
        double Matching ::getConfidence()
        {
            return confidenceCheck;
        }
        //!Method responsible for generating the disparity map
        void Matching::dispartyMapFormation(const Mat &costVolume, Mat &mapFinal, int th)
        {
            mapFinal.setTo(0);
            //uint8_t *map = mapFinal.data;
            int disparity = maxDisparity;
            //int width = costVolume.cols / ( disparity + 1) - 1;
            int height = costVolume.rows - 1;
            //memset(map, 0, sizeof(map[0]) * width * height);
            parallel_for_(Range(0,height - 1), makeMap(costVolume,th,disparity,confidenceCheck,scallingFactor,mapFinal));
        }
        //!1x9 median filter
        void Matching::Median1x9Filter(const Mat &originalImage, Mat &filteredImage)
        {
            CV_Assert(originalImage.rows == filteredImage.rows);
            CV_Assert(originalImage.cols == filteredImage.cols);
            parallel_for_(Range(1,originalImage.rows - 2), Median1x9(originalImage,filteredImage));
        }
        //!9x1 median filter
        void Matching::Median9x1Filter(const Mat &originalImage, Mat &filteredImage)
        {
            CV_Assert(originalImage.cols == filteredImage.cols);
            CV_Assert(originalImage.cols == filteredImage.cols);
            parallel_for_(Range(1,originalImage.cols - 2), Median9x1(originalImage,filteredImage));
        }
    }
}
