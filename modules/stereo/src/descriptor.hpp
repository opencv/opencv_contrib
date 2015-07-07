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
*   The interface contains the main descriptors that will be implemented in the descriptor class                  *
\******************************************************************************************************************/

#include "precomp.hpp"
#include <stdint.h>
#ifndef _OPENCV_DESCRIPTOR_HPP_
#define _OPENCV_DESCRIPTOR_HPP_
#ifdef __cplusplus

namespace cv
{
    namespace stereo
    {
        enum { Dense_Census, Sparse_Census, StarCensus};
        enum {ClassicCenterSymetricCensus, ModifiedCenterSymetricCensus};
        enum {StandardMct,MeanVariation};
        enum {SSE, NonSSE};
        //!Mean Variation is a robust kernel that compares a pixel
        //!not just with the center but also with the mean of the window
        struct MVKernel
        {
            uint8_t *image1;
            uint8_t *image2;
            uint8_t *integralLeft;
            uint8_t *integralRight;
            MVKernel(uint8_t *img, uint8_t *img2, uint8_t *integralL, uint8_t *integralR): image1(img),image2(img2),integralLeft(integralL), integralRight(integralR){}
            void operator()(int rrWidth,int w2, int rWidth, int jj, int j, int &c, int &c2) const
            {

            }
        };
        //!kernel that takes the pixels from certain positions from a patch
        //!offers verry good results
        struct StarKernel
        {
            uint8_t *image1;
            uint8_t *image2;
            StarKernel(uint8_t *img, uint8_t *img2): image1(img),image2(img2){}
            void operator()(int rrWidth,int w2, int rWidth, int jj, int j, int &c, int &c2) const
            {

            }
        };
        //!Compares pixels from a patch giving high weights to pixels in which
        //!the intensity is higher. The other pixels receive a lower weight
        struct MCTKernel
        {
            uint8_t *image1;
            uint8_t *image2;
            int t;
            MCTKernel(uint8_t * img,uint8_t *img2, int threshold) : image1(img),image2(img2), t(threshold) {}
            void operator()(int rrWidth,int w2, int rWidth, int jj, int j, int &c, int &c2) const
            {
                if (image1[rrWidth + jj] > image1[rWidth + j] - t)
                {
                    c <<= 2;
                    c |= 0x3;
                }
                else if (image1[rWidth + j] - t < image1[rrWidth + jj] && image1[rWidth + j] + t >= image1[rrWidth + jj])
                {
                    c <<= 2;
                    c = c + 1;
                }
                else
                {
                    c <<= 2;
                }
                if (image2[rrWidth + jj] > image2[rWidth + j] - t)
                {
                    c2 <<= 2;
                    c2 |= 0x3;
                }
                else if (image2[rWidth + j] - t < image2[rrWidth + jj] && image2[rWidth + j] + t >= image2[rrWidth + jj])
                {
                    c2 <<= 2;
                    c2 = c2 + 1;
                }
                else
                {
                    c2 <<= 2;
                }
            }
        };
        //!A madified cs census that compares a pixel with the imediat neightbour starting
        //!from the center
        struct ModifiedCsCensus
        {
            uint8_t *image1;
            uint8_t *image2;
            int n2;
            ModifiedCsCensus(uint8_t *im1, uint8_t *im2, int ker):image1(im1),image2(im2),n2(ker){}
            void operator()(int rrWidth,int w2, int rWidth, int jj, int j, int &c, int &c2) const
            {
                if (image1[(rrWidth + jj)] > image1[(w2 + (jj + n2))])
                {
                    c = c + 1;
                }
                c = c * 2;
                if (image2[(rrWidth + jj)] > image2[(w2 + (jj + n2))])
                {
                    c2 = c2 + 1;
                }
                c2 = c2 * 2;
            }
        };
        //!A kernel in which a pixel is compared with the center of the window
        struct CensusKernel
        {
            uint8_t *image1;
            uint8_t *image2;
            CensusKernel(uint8_t *im1, uint8_t *im2):image1(im1),image2(im2){}
            void operator()(int rrWidth,int w2, int rWidth, int jj, int j, int &c, int &c2) const
            {
                //compare a pixel with the center from the kernel
                if (image1[rrWidth + jj] > image1[rWidth + j])
                {
                    c = c + 1;
                }
                c = c * 2;
                //compare pixel with center for image 2
                if (image2[rrWidth + jj] > image2[rWidth + j])
                {
                    c2 = c2 + 1;
                }
                c2 = c2 * 2;
            }
        };

        //template clas which efficiently combines the descriptors
        template <int step_start, int step_end, int step_inc, typename Kernel>
        class CombinedDescriptor:public ParallelLoopBody
        {
        private:
            uint8_t *image1, *image2;
            int *dst1, *dst2;
            int n2 , width, height;
            int n2_stop;
            Kernel kernel_;
        public:
            CombinedDescriptor(int w, int h, int k2, int * distance1, int * distance2, Kernel kernel,int k2Stop) :
                width(w), height(h), n2(k2),dst1(distance1), dst2(distance2), kernel_(kernel), n2_stop(k2Stop){}
            void operator()(const cv::Range &r) const {
                for (int i = r.start; i <= r.end ; i++)
                {
                    int rWidth = i * width;
                    for (int j = n2 + 2; j <= width - n2 - 2; j++)
                    {
                        int c = 0;
                        int c2 = 0;
                        for(int step = step_start; step <= step_end; step += step_inc)
                        {
                            for (int ii = - n2; ii <= + n2_stop; ii += step)
                            {
                                int rrWidth = (ii + i) * width;
                                int rrWidthC = (ii + i + n2) * width;
                                for (int jj = j - n2; jj <= j + n2; jj += step)
                                {
                                    if (ii != i || jj != j)
                                    {
                                        kernel_(rrWidth,rrWidthC, rWidth, jj, j, c,c2);
                                    }
                                }
                            }
                        }
                        dst1[rWidth + j] = c;
                        dst2[rWidth + j] = c2;
                    }
                }
            }
        };
        //!class that implemented the census descriptor on single images
        class singleImageCensus : public ParallelLoopBody
        {
        private:
            uint8_t *image;
            int *dst;
            int n2, width, height, type;
        public:
            singleImageCensus(uint8_t * img1, int w, int h, int k2, int * distance1,const int t) :
                image(img1), dst(distance1), n2(k2), width(w), height(h), type(t){}
            void operator()(const cv::Range &r) const {
                for (int i = r.start; i <= r.end ; i++)
                {
                    int rWidth = i * width;
                    for (int j = n2; j <= width - n2; j++)
                    {
                        if (type == SSE)
                        {
                            //to do
                        }
                        else
                        {
                            int c = 0;
                            for (int ii = i - n2; ii <= i + n2; ii++)
                            {
                                int rrWidth = ii * width;
                                for (int jj = j - n2; jj <= j + n2; jj++)
                                {
                                    if (ii != i || jj != j)
                                    {
                                        if (image[(rrWidth + jj)] > image[(rWidth + j)])
                                        {
                                            c = c + 1;
                                        }
                                        c = c * 2;
                                    }
                                }
                            }
                            dst[(rWidth + j)] = c;
                        }
                    }
                }
            }
        };
        //!paralel implementation of the center symetric census
        class parallelSymetricCensus:public ParallelLoopBody
        {
        private:
            uint8_t *image1, *image2;
            int *dst1, *dst2;
            int n2, width, height, type;
        public:
            parallelSymetricCensus(uint8_t * img1, uint8_t * img2, int w, int h, int k2, int * distance1, int * distance2,const int t) :
                image1(img1), image2(img2), dst1(distance1), dst2(distance2), n2(k2), width(w), height(h), type(t){}
            void operator()(const cv::Range &r) const {
                for (int i = r.start; i <= r.end ; i++)
                {
                    int distV = (i)* width;
                    for (int j = n2; j <= width - n2; j++)
                    {
                        int c = 0;
                        int c2 = 0;
                        //the classic center symetric census which compares the curent pixel with its symetric not its center.
                        for (int ii = -n2; ii < 0; ii++)
                        {
                            int rrWidth = (ii + i) * width;
                            for (int jj = -n2; jj <= +n2; jj++)
                            {
                                if (image1[(rrWidth + (jj + j))] > image1[((ii * (-1) + i) * width + (-1 * jj) + j)])
                                {
                                    c = c + 1;
                                }
                                c = c * 2;

                                if (image2[(rrWidth + (jj + j))] > image2[((ii * (-1) + i) * width + (-1 * jj) + j)])
                                {
                                    c2 = c2 + 1;
                                }
                                c2 = c2 * 2;
                            }
                        }
                        for (int jj = -n2; jj < 0; jj++)
                        {
                            if (image1[(i * width + (jj + j))] > image1[(i * width + (-1 * jj) + j)])
                            {
                                c = c + 1;
                            }
                            c = c * 2;
                            if (image2[(i * width + (jj + j))] > image2[(i * width + (-1 * jj) + j)])
                            {
                                c2 = c2 + 1;
                            }
                            c2 = c2 * 2;
                        }//a modified version of cs census which compares each pixel with its correspondent from
                        //the same distance from the center
                        dst1[(distV + j)] = c;
                        dst2[(distV + j)] = c2;
                    }
                }
            }
        };
        //!Implementation for computing the Census transform on the given image
        void applyCensusOnImage(const cv::Mat &img, int kernelSize, cv::Mat &dist, const int type);
        /**
        Two variations of census applied on input images
        Implementation of a census transform which is taking into account just the some pixels from the census kernel thus allowing for larger block sizes
        **/
        void applyCensusOnImages(const cv::Mat &im1,const cv::Mat &im2, int kernelSize, cv::Mat &dist, cv::Mat &dist2, const int type);
        /**
        STANDARD_MCT - Modified census which is memorizing for each pixel 2 bits and includes a tolerance to the pixel comparison
        MCT_MEAN_VARIATION - Implementation of a modified census transform which is also taking into account the variation to the mean of the window not just the center pixel
        **/
        void applyMCTOnImages(const cv::Mat &img1, const cv::Mat &img2, int kernelSize, int t, cv::Mat &dist, cv::Mat &dist2, const int type);
        /**The classical center symetric census
        A modified version of cs census which is comparing a pixel with its correspondent after the center
        **/
        void applySimetricCensus(const cv::Mat &img1, const cv::Mat &img2, int kernelSize, cv::Mat &dist, cv::Mat &dist2, const int type);
        //!brief binary descriptor used in stereo correspondence
        void applyBrifeDescriptor(const cv::Mat &image1, const cv::Mat &image2, int kernelSize, cv::Mat &dist, cv::Mat &dist2);
        //The classical Rank Transform
        void  applyRTDescriptor(const cv::Mat &image1, const cv::Mat &image2, int kernelSize, cv::Mat &dist, cv::Mat &dist2);
    }
}
#endif
#endif
/*End of file*/
