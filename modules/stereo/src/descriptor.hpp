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
                CV_UNUSED(w2);
                for (int i = 0; i < stop; i++)
                {
                    if (image[i][rrWidth + jj] > image[i][rWidth + j])
                    {
                        c[i] += 1;
                    }
                    c[i] <<= 1;
                    if (integralImage[i][rrWidth + jj] > image[i][rWidth + j])
                    {
                        c[i] += 1;
                    }
                    c[i] <<= 1;
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
                CV_UNUSED(w2);
                for(int i = 0; i < imageStop; i++)
                {
                    c[i] <<= 2;
                    if (image[i][rrWidth + jj] > image[i][rWidth + j] + t)
                        c[i] += 3;
                    else if (image[i][rrWidth + jj] > image[i][rWidth + j] - t)
                        c[i] += 1;
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
                CV_UNUSED(j);
                CV_UNUSED(rWidth);
                for(int i = 0; i < imageStop; i++)
                {
                    if (image[i][(rrWidth + jj)] > image[i][(w2 + (jj + n2))])
                    {
                        c[i] += 1;
                    }
                    c[i] <<= 1;
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
                CV_UNUSED(w2);
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
                for (int i = r.start; i < r.end ; i++)
                {
                    int rWidth = i * stride_;
                    for (int j = 0; j < width; j++)
                    {
                        if (i < n2 || i >= height - n2 || j < n2 + 2 || j >= width - n2 - 2)
                        {
                            for(int l = 0; l < nr_img; l++)
                                dst[l][rWidth + j] = 0;  // TODO out of range value?
                            continue;
                        }

                        int c[nr_img];
                        memset(c, 0, sizeof(c[0]) * nr_img);
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
                for (int i = r.start; i < r.end; i++)
                {
                    int rWidth = i * stride_;
                    for (int j = 0; j < width; j++)
                    {
                        for(int d = 0 ; d < im_num; d++)
                        {
                            if (i < n2 || i >= height - n2 || j < n2 || j >= width - n2)
                            {
                                dst[d][rWidth + j] = 0;  // TODO out of range value?
                                continue;
                            }
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
                for (int i = r.start; i < r.end ; i++)
                {
                    int distV = i*stride_;
                    for (int j = 0; j < width; j++)
                    {
                        for(int d = 0; d < im_num; d++)
                        {
                            if (i < n2 || i >= height - n2 || j < n2 || j >= width - n2)
                            {
                                dst[d][distV + j] = 0;  // TODO out of range value?
                                continue;
                            }
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
    }
}
#endif
#endif
/*End of file*/
