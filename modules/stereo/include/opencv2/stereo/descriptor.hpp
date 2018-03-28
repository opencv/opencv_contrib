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
\*****************************************************************************************************************/

#include <stdint.h>
#ifndef _OPENCV_DESCRIPTOR_HPP_
#define _OPENCV_DESCRIPTOR_HPP_
#ifdef __cplusplus

namespace cv
{
    namespace stereo
    {
        //types of supported kernels
        enum {
            CV_DENSE_CENSUS, CV_SPARSE_CENSUS,
            CV_CS_CENSUS, CV_MODIFIED_CS_CENSUS, CV_MODIFIED_CENSUS_TRANSFORM,
            CV_MEAN_VARIATION, CV_STAR_KERNEL
        };
        //!Mean Variation is a robust kernel that compares a pixel
        //!not just with the center but also with the mean of the window
        template<int num_images>
        struct MVKernel
        {
            uint8_t *image[num_images];
            int *integralImage[num_images];
            int stop;
            MVKernel(){}
            MVKernel(uint8_t **images, int **integral)
            {
                for(int i = 0; i < num_images; i++)
                {
                    image[i] = images[i];
                    integralImage[i] = integral[i];
                }
                stop = num_images;
            }
            void operator()(int rrWidth,int w2, int rWidth, int jj, int j, int c[num_images]) const
            {
                (void)w2;
                for (int i = 0; i < stop; i++)
                {
                    if (image[i][rrWidth + jj] > image[i][rWidth + j])
                    {
                        c[i] = c[i] + 1;
                    }
                    c[i] = c[i] << 1;
                    if (integralImage[i][rrWidth + jj] > image[i][rWidth + j])
                    {
                        c[i] = c[i] + 1;
                    }
                    c[i] = c[i] << 1;
                }
            }
        };
        //!Compares pixels from a patch giving high weights to pixels in which
        //!the intensity is higher. The other pixels receive a lower weight
        template <int num_images>
        struct MCTKernel
        {
            uint8_t *image[num_images];
            int t,imageStop;
            MCTKernel(){}
            MCTKernel(uint8_t ** images, int threshold)
            {
                for(int i = 0; i < num_images; i++)
                {
                    image[i] = images[i];
                }
                imageStop = num_images;
                t = threshold;
            }
            void operator()(int rrWidth,int w2, int rWidth, int jj, int j, int c[num_images]) const
            {
                (void)w2;
                for(int i = 0; i < imageStop; i++)
                {
                    if (image[i][rrWidth + jj] > image[i][rWidth + j] - t)
                    {
                        c[i] = c[i] << 1;
                        c[i] = c[i] + 1;
                        c[i] = c[i] << 1;
                        c[i] = c[i] + 1;
                    }
                    else if (image[i][rWidth + j] - t < image[i][rrWidth + jj] && image[i][rWidth + j] + t >= image[i][rrWidth + jj])
                    {
                        c[i] = c[i] << 2;
                        c[i] = c[i] + 1;
                    }
                    else
                    {
                        c[i] <<= 2;
                    }
                }
            }
        };
        //!A madified cs census that compares a pixel with the imediat neightbour starting
        //!from the center
        template<int num_images>
        struct ModifiedCsCensus
        {
            uint8_t *image[num_images];
            int n2;
            int imageStop;
            ModifiedCsCensus(){}
            ModifiedCsCensus(uint8_t **images, int ker)
            {
                for(int i = 0; i < num_images; i++)
                    image[i] = images[i];
                imageStop = num_images;
                n2 = ker;
            }
            void operator()(int rrWidth,int w2, int rWidth, int jj, int j, int c[num_images]) const
            {
                (void)j;
                (void)rWidth;
                for(int i = 0; i < imageStop; i++)
                {
                    if (image[i][(rrWidth + jj)] > image[i][(w2 + (jj + n2))])
                    {
                        c[i] = c[i] + 1;
                    }
                    c[i] = c[i] * 2;
                }
            }
        };
        //!A kernel in which a pixel is compared with the center of the window
        template<int num_images>
        struct CensusKernel
        {
            uint8_t *image[num_images];
            int imageStop;
            CensusKernel(){}
            CensusKernel(uint8_t **images)
            {
                for(int i = 0; i < num_images; i++)
                    image[i] = images[i];
                imageStop = num_images;
            }
            void operator()(int rrWidth,int w2, int rWidth, int jj, int j, int c[num_images]) const
            {
                (void)w2;
                for(int i = 0; i < imageStop; i++)
                {
                    ////compare a pixel with the center from the kernel
                    if (image[i][rrWidth + jj] > image[i][rWidth + j])
                    {
                        c[i] += 1;
                    }
                    c[i] <<= 1;
                }
            }
        };
        //template clas which efficiently combines the descriptors
        template <int step_start, int step_end, int step_inc,int nr_img, typename Kernel>
        class CombinedDescriptor:public ParallelLoopBody
        {
        private:
            int width, height,n2;
            int stride_;
            int *dst[nr_img];
            Kernel kernel_;
            int n2_stop;
        public:
            CombinedDescriptor(int w, int h,int stride, int k2, int **distance, Kernel kernel,int k2Stop)
            {
                width = w;
                height = h;
                n2 = k2;
                stride_ = stride;
                for(int i = 0; i < nr_img; i++)
                    dst[i] = distance[i];
                kernel_ = kernel;
                n2_stop = k2Stop;
            }
            void operator()(const cv::Range &r) const CV_OVERRIDE {
                for (int i = r.start; i <= r.end ; i++)
                {
                    int rWidth = i * stride_;
                    for (int j = n2 + 2; j <= width - n2 - 2; j++)
                    {
                        int c[nr_img];
                        memset(c,0,nr_img);
                        for(int step = step_start; step <= step_end; step += step_inc)
                        {
                            for (int ii = - n2; ii <= + n2_stop; ii += step)
                            {
                                int rrWidth = (ii + i) * stride_;
                                int rrWidthC = (ii + i + n2) * stride_;
                                for (int jj = j - n2; jj <= j + n2; jj += step)
                                {
                                    if (ii != i || jj != j)
                                    {
                                        kernel_(rrWidth,rrWidthC, rWidth, jj, j,c);
                                    }
                                }
                            }
                        }
                        for(int l = 0; l < nr_img; l++)
                            dst[l][rWidth + j] = c[l];
                    }
                }
            }
        };
        //!calculate the mean of every windowSizexWindwoSize block from the integral Image
        //!this is a preprocessing for MV kernel
        class MeanKernelIntegralImage : public ParallelLoopBody
        {
        private:
            int *img;
            int windowSize,width;
            float scalling;
            int *c;
        public:
            MeanKernelIntegralImage(const cv::Mat &image, int window,float scale, int *cost):
                img((int *)image.data),windowSize(window) ,width(image.cols) ,scalling(scale) , c(cost){};
            void operator()(const cv::Range &r) const CV_OVERRIDE {
                for (int i = r.start; i <= r.end; i++)
                {
                    int iw = i * width;
                    for (int j = windowSize + 1; j <= width - windowSize - 1; j++)
                    {
                        c[iw + j] = (int)((img[(i + windowSize - 1) * width + j + windowSize - 1] + img[(i - windowSize - 1) * width + j - windowSize - 1]
                        - img[(i + windowSize) * width + j - windowSize] - img[(i - windowSize) * width + j + windowSize]) * scalling);
                    }
                }
            }
        };
        //!implementation for the star kernel descriptor
        template<int num_images>
        class StarKernelCensus:public ParallelLoopBody
        {
        private:
            uint8_t *image[num_images];
            int *dst[num_images];
            int n2, width, height, im_num,stride_;
        public:
            StarKernelCensus(const cv::Mat *img, int k2, int **distance)
            {
                for(int i = 0; i < num_images; i++)
                {
                    image[i] = img[i].data;
                    dst[i] = distance[i];
                }
                n2 = k2;
                width = img[0].cols;
                height = img[0].rows;
                im_num = num_images;
                stride_ = (int)img[0].step;
            }
            void operator()(const cv::Range &r) const CV_OVERRIDE {
                for (int i = r.start; i <= r.end ; i++)
                {
                    int rWidth = i * stride_;
                    for (int j = n2; j <= width - n2; j++)
                    {
                        for(int d = 0 ; d < im_num; d++)
                        {
                            int c = 0;
                            for(int step = 4; step > 0; step--)
                            {
                                for (int ii = i - step; ii <= i + step; ii += step)
                                {
                                    int rrWidth = ii * stride_;
                                    for (int jj = j - step; jj <= j + step; jj += step)
                                    {
                                        if (image[d][rrWidth + jj] > image[d][rWidth + j])
                                        {
                                            c = c + 1;
                                        }
                                        c = c * 2;
                                    }
                                }
                            }
                            for (int ii = -1; ii <= +1; ii++)
                            {
                                int rrWidth = (ii + i) * stride_;
                                if (i == -1)
                                {
                                    if (ii + i != i)
                                    {
                                        if (image[d][rrWidth + j] > image[d][rWidth + j])
                                        {
                                            c = c + 1;
                                        }
                                        c = c * 2;
                                    }
                                }
                                else if (i == 0)
                                {
                                    for (int j2 = -1; j2 <= 1; j2 += 2)
                                    {
                                        if (ii + i != i)
                                        {
                                            if (image[d][rrWidth + j + j2] > image[d][rWidth + j])
                                            {
                                                c = c + 1;
                                            }
                                            c = c * 2;
                                        }
                                    }
                                }
                                else
                                {
                                    if (ii + i != i)
                                    {
                                        if (image[d][rrWidth + j] > image[d][rWidth + j])
                                        {
                                            c = c + 1;
                                        }
                                        c = c * 2;
                                    }
                                }
                            }
                            dst[d][rWidth + j] = c;
                        }
                    }
                }
            }
        };
        //!paralel implementation of the center symetric census
        template <int num_images>
        class SymetricCensus:public ParallelLoopBody
        {
        private:
            uint8_t *image[num_images];
            int *dst[num_images];
            int n2, width, height, im_num,stride_;
        public:
            SymetricCensus(const cv::Mat *img, int k2, int **distance)
            {
                for(int i = 0; i < num_images; i++)
                {
                    image[i] = img[i].data;
                    dst[i] = distance[i];
                }
                n2 = k2;
                width = img[0].cols;
                height = img[0].rows;
                im_num = num_images;
                stride_ = (int)img[0].step;
            }
            void operator()(const cv::Range &r) const CV_OVERRIDE {
                for (int i = r.start; i <= r.end ; i++)
                {
                    int distV = i*stride_;
                    for (int j = n2; j <= width - n2; j++)
                    {
                        for(int d = 0; d < im_num; d++)
                        {
                            int c = 0;
                            //the classic center symetric census which compares the curent pixel with its symetric not its center.
                            for (int ii = -n2; ii <= 0; ii++)
                            {
                                int rrWidth = (ii + i) * stride_;
                                for (int jj = -n2; jj <= +n2; jj++)
                                {
                                    if (image[d][(rrWidth + (jj + j))] > image[d][((ii * (-1) + i) * width + (-1 * jj) + j)])
                                    {
                                        c = c + 1;
                                    }
                                    c = c * 2;
                                    if(ii == 0 && jj < 0)
                                    {
                                        if (image[d][(i * width + (jj + j))] > image[d][(i * width + (-1 * jj) + j)])
                                        {
                                            c = c + 1;
                                        }
                                        c = c * 2;
                                    }
                                }
                            }
                            dst[d][(distV + j)] = c;
                        }
                    }
                }
            }
        };
        /**
        Two variations of census applied on input images
        Implementation of a census transform which is taking into account just the some pixels from the census kernel thus allowing for larger block sizes
        **/
        //void applyCensusOnImages(const cv::Mat &im1,const cv::Mat &im2, int kernelSize, cv::Mat &dist, cv::Mat &dist2, const int type);
        CV_EXPORTS void censusTransform(const cv::Mat &image1, const cv::Mat &image2, int kernelSize, cv::Mat &dist1, cv::Mat &dist2, const int type);
        //single image census transform
        CV_EXPORTS void censusTransform(const cv::Mat &image1, int kernelSize, cv::Mat &dist1, const int type);
        /**
        STANDARD_MCT - Modified census which is memorizing for each pixel 2 bits and includes a tolerance to the pixel comparison
        MCT_MEAN_VARIATION - Implementation of a modified census transform which is also taking into account the variation to the mean of the window not just the center pixel
        **/
        CV_EXPORTS void modifiedCensusTransform(const cv::Mat &img1, const cv::Mat &img2, int kernelSize, cv::Mat &dist1,cv::Mat &dist2, const int type, int t = 0 , const cv::Mat &IntegralImage1 = cv::Mat::zeros(100,100,CV_8UC1), const cv::Mat &IntegralImage2 = cv::Mat::zeros(100,100,CV_8UC1));
        //single version of modified census transform descriptor
        CV_EXPORTS void modifiedCensusTransform(const cv::Mat &img1, int kernelSize, cv::Mat &dist, const int type, int t = 0 ,const cv::Mat &IntegralImage = cv::Mat::zeros(100,100,CV_8UC1));
        /**The classical center symetric census
        A modified version of cs census which is comparing a pixel with its correspondent after the center
        **/
        CV_EXPORTS void symetricCensusTransform(const cv::Mat &img1, const cv::Mat &img2, int kernelSize, cv::Mat &dist1, cv::Mat &dist2, const int type);
        //single version of census transform
        CV_EXPORTS void symetricCensusTransform(const cv::Mat &img1, int kernelSize, cv::Mat &dist1, const int type);
        //in a 9x9 kernel only certain positions are choosen
        CV_EXPORTS void starCensusTransform(const cv::Mat &img1, const cv::Mat &img2, int kernelSize, cv::Mat &dist1,cv::Mat &dist2);
        //single image version of star kernel
        CV_EXPORTS void starCensusTransform(const cv::Mat &img1, int kernelSize, cv::Mat &dist);
        //integral image computation used in the Mean Variation Census Transform
        void imageMeanKernelSize(const cv::Mat &img, int windowSize, cv::Mat &c);
    }
}
#endif
#endif
/*End of file*/
