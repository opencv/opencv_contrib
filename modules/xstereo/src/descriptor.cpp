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
*                             The file contains the implemented descriptors                                       *
\******************************************************************************************************************/
#include "precomp.hpp"

namespace cv
{
    namespace stereo
    {
        //function that performs the census transform on two images.
        //Two variants of census are offered a sparse version whcih takes every second pixel as well as dense version
        CV_EXPORTS void censusTransform(const Mat &image1, const Mat &image2, int kernelSize, Mat &dist1, Mat &dist2, const int type)
        {
            CV_Assert(image1.size() == image2.size());
            CV_Assert(kernelSize % 2 != 0);
            CV_Assert(image1.type() == CV_8UC1 && image2.type() == CV_8UC1);
            CV_Assert(type == CV_DENSE_CENSUS || type == CV_SPARSE_CENSUS);
            CV_Assert(kernelSize <= ((type == 0) ? 5 : 11));
            int n2 = (kernelSize) / 2;
            uint8_t *images[] = {image1.data, image2.data};
            int *costs[] = {(int *)dist1.data,(int *)dist2.data};
            int stride = (int)image1.step;
            if(type == CV_DENSE_CENSUS)
            {
                parallel_for_(Range(0, image1.rows),
                    CombinedDescriptor<1,1,1,2,CensusKernel<2> >(image1.cols, image1.rows,stride,n2,costs,CensusKernel<2>(images),n2));
            }
            else if(type == CV_SPARSE_CENSUS)
            {
                parallel_for_(Range(0, image1.rows),
                    CombinedDescriptor<2,2,1,2,CensusKernel<2> >(image1.cols, image1.rows, stride,n2,costs,CensusKernel<2>(images),n2));
            }
        }
        //function that performs census on one image
        CV_EXPORTS void censusTransform(const Mat &image1, int kernelSize, Mat &dist1, const int type)
        {
            CV_Assert(image1.size() == dist1.size());
            CV_Assert(kernelSize % 2 != 0);
            CV_Assert(image1.type() == CV_8UC1);
            CV_Assert(type == CV_DENSE_CENSUS || type == CV_SPARSE_CENSUS);
            CV_Assert(kernelSize <= ((type == 0) ? 5 : 11));
            int n2 = (kernelSize) / 2;
            uint8_t *images[] = {image1.data};
            int *costs[] = {(int *)dist1.data};
            int stride = (int)image1.step;
            if(type == CV_DENSE_CENSUS)
            {
                parallel_for_(Range(0, image1.rows),
                    CombinedDescriptor<1,1,1,1,CensusKernel<1> >(image1.cols, image1.rows,stride,n2,costs,CensusKernel<1>(images),n2));
            }
            else if(type == CV_SPARSE_CENSUS)
            {
                parallel_for_(Range(0, image1.rows),
                    CombinedDescriptor<2,2,1,1,CensusKernel<1> >(image1.cols, image1.rows,stride,n2,costs,CensusKernel<1>(images),n2));
            }
        }
        //in a 9x9 kernel only certain positions are choosen for comparison
        CV_EXPORTS void starCensusTransform(const Mat &img1, const Mat &img2, int kernelSize, Mat &dist1, Mat &dist2)
        {
            CV_Assert(img1.size() == img2.size());
            CV_Assert(kernelSize % 2 != 0);
            CV_Assert(img1.type() == CV_8UC1 && img2.type() == CV_8UC1);
            CV_Assert(kernelSize >= 7);
            int n2 = (kernelSize) >> 1;
            Mat images[] = {img1, img2};
            int *date[] = { (int *)dist1.data, (int *)dist2.data};
            parallel_for_(Range(0, img1.rows), StarKernelCensus<2>(images, n2,date));
        }
        //single version of star census
        CV_EXPORTS void starCensusTransform(const Mat &img1, int kernelSize, Mat &dist)
        {
            CV_Assert(img1.size() == dist.size());
            CV_Assert(kernelSize % 2 != 0);
            CV_Assert(img1.type() == CV_8UC1);
            CV_Assert(kernelSize >= 7);
            int n2 = (kernelSize) >> 1;
            Mat images[] = {img1};
            int *date[] = { (int *)dist.data};
            parallel_for_(Range(0, img1.rows), StarKernelCensus<1>(images, n2,date));
        }
        //Modified census transforms
        //the first one deals with small illumination changes
        //the sencond modified census transform is invariant to noise; i.e.
        //if the current pixel with whom we are dooing the comparison is a noise, this descriptor will provide a better result by comparing with the mean of the window
        //otherwise if the pixel is not noise the information is strengthend
        CV_EXPORTS void modifiedCensusTransform(const Mat &img1, const Mat &img2, int kernelSize, Mat &dist1,Mat &dist2, const int type, int t, const Mat& integralImage1, const Mat& integralImage2)
        {
            CV_Assert(img1.size() == img2.size());
            CV_Assert(kernelSize % 2 != 0);
            CV_Assert(img1.type() == CV_8UC1 && img2.type() == CV_8UC1);
            CV_Assert(type == CV_MODIFIED_CENSUS_TRANSFORM || type == CV_MEAN_VARIATION);
            CV_Assert(kernelSize <= 9);
            int n2 = (kernelSize - 1) >> 1;
            uint8_t *images[] = {img1.data, img2.data};
            int *date[] = { (int *)dist1.data, (int *)dist2.data};
            int stride = (int)img1.cols;
            if(type == CV_MODIFIED_CENSUS_TRANSFORM)
            {
                //MCT
                parallel_for_(Range(0, img1.rows),
                    CombinedDescriptor<2,4,2, 2,MCTKernel<2> >(img1.cols, img1.rows,stride,n2,date,MCTKernel<2>(images,t),n2));
            }
            else if(type == CV_MEAN_VARIATION)
            {
                //MV
                CV_Assert(!integralImage1.empty());
                CV_Assert(!integralImage1.isContinuous());
                CV_CheckTypeEQ(integralImage1.type(), CV_32SC1, "");
                CV_CheckGE(integralImage1.cols, img1.cols, "");
                CV_CheckGE(integralImage1.rows, img1.rows, "");
                CV_Assert(!integralImage2.empty());
                CV_Assert(!integralImage2.isContinuous());
                CV_CheckTypeEQ(integralImage2.type(), CV_32SC1, "");
                CV_CheckGE(integralImage2.cols, img2.cols, "");
                CV_CheckGE(integralImage2.rows, img2.rows, "");
                int *integral[2] = {
                        (int *)integralImage1.data,
                        (int *)integralImage2.data
                };
                parallel_for_(Range(0, img1.rows),
                    CombinedDescriptor<2,3,2,2, MVKernel<2> >(img1.cols, img1.rows,stride,n2,date,MVKernel<2>(images,integral),n2));
            }
        }
        CV_EXPORTS void modifiedCensusTransform(const Mat &img1, int kernelSize, Mat &dist, const int type, int t , Mat const &integralImage)
        {
            CV_Assert(img1.size() == dist.size());
            CV_Assert(kernelSize % 2 != 0);
            CV_Assert(img1.type() == CV_8UC1);
            CV_Assert(type == CV_MODIFIED_CENSUS_TRANSFORM || type == CV_MEAN_VARIATION);
            CV_Assert(kernelSize <= 9);
            int n2 = (kernelSize - 1) >> 1;
            uint8_t *images[] = {img1.data};
            int *date[] = { (int *)dist.data};
            int stride = (int)img1.step;
            if(type == CV_MODIFIED_CENSUS_TRANSFORM)
            {
                //MCT
                parallel_for_(Range(0, img1.rows),
                    CombinedDescriptor<2,4,2, 1,MCTKernel<1> >(img1.cols, img1.rows,stride,n2,date,MCTKernel<1>(images,t),n2));
            }
            else if(type == CV_MEAN_VARIATION)
            {
                //MV
                CV_Assert(!integralImage.empty());
                CV_Assert(!integralImage.isContinuous());
                CV_CheckTypeEQ(integralImage.type(), CV_32SC1, "");
                CV_CheckGE(integralImage.cols, img1.cols, "");
                CV_CheckGE(integralImage.rows, img1.rows, "");
                int *integral[] = { (int *)integralImage.data};
                parallel_for_(Range(0, img1.rows),
                    CombinedDescriptor<2,3,2,1, MVKernel<1> >(img1.cols, img1.rows,stride,n2,date,MVKernel<1>(images,integral),n2));
            }
        }
        //different versions of simetric census
        //These variants since they do not compare with the center they are invariant to noise
        CV_EXPORTS void symetricCensusTransform(const Mat &img1, const Mat &img2, int kernelSize, Mat &dist1, Mat &dist2, const int type)
        {
            CV_Assert(img1.size() ==  img2.size());
            CV_Assert(kernelSize % 2 != 0);
            CV_Assert(img1.type() == CV_8UC1 && img2.type() == CV_8UC1);
            CV_Assert(type == CV_CS_CENSUS || type == CV_MODIFIED_CS_CENSUS);
            CV_Assert(kernelSize <= 7);
            int n2 = kernelSize >> 1;
            uint8_t *images[] = {img1.data, img2.data};
            Mat imag[] = {img1, img2};
            int *date[] = { (int *)dist1.data, (int *)dist2.data};
            int stride = (int)img1.step;
            if(type == CV_CS_CENSUS)
            {
                parallel_for_(Range(0, img1.rows), SymetricCensus<2>(imag, n2,date));
            }
            else if(type == CV_MODIFIED_CS_CENSUS)
            {
                parallel_for_(Range(0, img1.rows),
                    CombinedDescriptor<1,1,1,2,ModifiedCsCensus<2> >(img1.cols, img1.rows,stride,n2,date,ModifiedCsCensus<2>(images,n2),1));
            }
        }
        CV_EXPORTS void symetricCensusTransform(const Mat &img1, int kernelSize, Mat &dist1, const int type)
        {
            CV_Assert(img1.size() ==  dist1.size());
            CV_Assert(kernelSize % 2 != 0);
            CV_Assert(img1.type() == CV_8UC1);
            CV_Assert(type == CV_MODIFIED_CS_CENSUS || type == CV_CS_CENSUS);
            CV_Assert(kernelSize <= 7);
            int n2 = kernelSize >> 1;
            uint8_t *images[] = {img1.data};
            Mat imag[] = {img1};
            int *date[] = { (int *)dist1.data};
            int stride = (int)img1.step;
            if(type == CV_CS_CENSUS)
            {
                parallel_for_(Range(0, img1.rows), SymetricCensus<1>(imag, n2,date));
            }
            else if(type == CV_MODIFIED_CS_CENSUS)
            {
                parallel_for_( Range(0, img1.rows),
                    CombinedDescriptor<1,1,1,1,ModifiedCsCensus<1> >(img1.cols, img1.rows,stride,n2,date,ModifiedCsCensus<1>(images,n2),1));
            }
        }
    }
}
