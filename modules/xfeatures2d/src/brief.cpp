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
// Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"
#include <algorithm>
#include <vector>

#include <iostream>
#include <iomanip>

namespace cv
{
namespace xfeatures2d
{

/*
 * BRIEF Descriptor
 */
class BriefDescriptorExtractorImpl : public BriefDescriptorExtractor
{
public:
    enum { PATCH_SIZE = 48, KERNEL_SIZE = 9 };

    // bytes is a length of descriptor in bytes. It can be equal 16, 32 or 64 bytes.
    BriefDescriptorExtractorImpl( int bytes = 32, bool use_orientation = false );

    virtual void read( const FileNode& );
    virtual void write( FileStorage& ) const;

    virtual int descriptorSize() const;
    virtual int descriptorType() const;
    virtual int defaultNorm() const;

    virtual void compute(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors);

protected:
    typedef void(*PixelTestFn)(InputArray, const std::vector<KeyPoint>&, OutputArray, bool use_orientation_);

    int bytes_;
    bool use_orientation_;
    PixelTestFn test_fn_;
};

Ptr<BriefDescriptorExtractor> BriefDescriptorExtractor::create( int bytes, bool use_orientation )
{
    return makePtr<BriefDescriptorExtractorImpl>(bytes, use_orientation);
}

inline int smoothedSum(const Mat& sum, const KeyPoint& pt, int y, int x)
{
    static const int HALF_KERNEL = BriefDescriptorExtractorImpl::KERNEL_SIZE / 2;

    int img_y = (int)(pt.pt.y + 0.5) + y;
    int img_x = (int)(pt.pt.x + 0.5) + x;
    return   sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1)
           - sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL)
           - sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1)
           + sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL);
}

static void calculateSums(const Mat &sum, const int count, const uchar *desc, const float cos_theta, const float sin_theta, KeyPoint pt, int &suma, int &sumb)
{
    int ax = desc[count];
    int ay = desc[count + 1];

    int bx = desc[count + 2];
    int by = desc[count + 3];

    int ax2 = (int) (((float)ax)*cos_theta - ((float)ay)*sin_theta);
    int ay2 = (int) (((float)ax)*sin_theta + ((float)ay)*cos_theta);
    int bx2 = (int) (((float)bx)*cos_theta - ((float)by)*sin_theta);
    int by2 = (int) (((float)bx)*sin_theta + ((float)by)*cos_theta);

    int half_patch_size = BriefDescriptorExtractorImpl::PATCH_SIZE/2;
    if (ax2 > half_patch_size)
      ax2 = half_patch_size;
    if (ax2 < -half_patch_size)
      ax2 = -half_patch_size;

    if (ay2 > half_patch_size)
      ay2 = half_patch_size;
    if (ay2 < -half_patch_size)
      ay2 = -half_patch_size;

    if (bx2 > half_patch_size)
      bx2 = half_patch_size;
    if (bx2 < -half_patch_size)
      bx2 = -half_patch_size;

    if (by2 > half_patch_size)
      by2 = half_patch_size;
    if (by2 < -half_patch_size)
      by2 = -half_patch_size;

    suma = smoothedSum(sum, pt, ay2, ax2);
    sumb = smoothedSum(sum, pt, by2, bx2);
}

static void pixelTests16(InputArray _sum, const std::vector<KeyPoint>& keypoints, OutputArray _descriptors, bool use_orientation)
{
    Mat sum = _sum.getMat(), descriptors = _descriptors.getMat();
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        uchar* desc = descriptors.ptr(static_cast<int>(i));
        const KeyPoint& pt = keypoints[i];
#include "generated_16.i"
        // invariance routine
        if ( use_orientation )
        {
            float angle = pt.angle;
            angle *= (float)(CV_PI / 180.f);
            float cos_theta = cos(angle);
            float sin_theta = sin(angle);

            int count = 0;
            for (int ix = 0; ix < 16; ix++){
                for (int jx = 7; jx >= 0; jx--){
                    int suma, sumb;
                    calculateSums(sum, count, desc, cos_theta, sin_theta, pt, suma, sumb);
                    desc[ix] += (uchar)((suma < sumb) << jx);
                    count += 4;
                }
            }
        }
    }
}

static void pixelTests32(InputArray _sum, const std::vector<KeyPoint>& keypoints, OutputArray _descriptors, bool use_orientation)
{
    Mat sum = _sum.getMat(), descriptors = _descriptors.getMat();
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        uchar* desc = descriptors.ptr(static_cast<int>(i));
        const KeyPoint& pt = keypoints[i];
#include "generated_32.i"
        // invariance routine
        if ( use_orientation )
        {
            float angle = pt.angle;
            angle *= (float)(CV_PI / 180.f);
            float cos_theta = cos(angle);
            float sin_theta = sin(angle);
            int count = 0;

            for (int ix = 0; ix < 32; ix++){
                for (int jx = 7; jx >= 0; jx--){
                    int suma, sumb;
                    calculateSums(sum, count, desc, cos_theta, sin_theta, pt, suma, sumb);
                    desc[ix] += (uchar)((suma < sumb) << jx);
                    count += 4;
                }
            }
        }
    }
}

static void pixelTests64(InputArray _sum, const std::vector<KeyPoint>& keypoints, OutputArray _descriptors, bool use_orientation)
{
    Mat sum = _sum.getMat(), descriptors = _descriptors.getMat();
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        uchar* desc = descriptors.ptr(static_cast<int>(i));
        const KeyPoint& pt = keypoints[i];
#include "generated_64.i"
        // invariance routine
        if ( use_orientation )
        {
            float angle = pt.angle;
            angle *= (float)(CV_PI / 180.f);
            float cos_theta = cos(angle);
            float sin_theta = sin(angle);

            int count = 0;
            for (int ix = 0; ix < 64; ix++){
                for (int jx = 7; jx >= 0; jx--){
                    int suma, sumb;
                    calculateSums(sum, count, desc, cos_theta, sin_theta, pt, suma, sumb);
                    desc[ix] += (uchar)((suma < sumb) << jx);
                    count += 4;
                }
            }
        }
    }
}

BriefDescriptorExtractorImpl::BriefDescriptorExtractorImpl(int bytes, bool use_orientation) :
    bytes_(bytes), test_fn_(NULL)
{
    use_orientation_ = use_orientation;

    switch (bytes)
    {
        case 16:
            test_fn_ = pixelTests16;
            break;
        case 32:
            test_fn_ = pixelTests32;
            break;
        case 64:
            test_fn_ = pixelTests64;
            break;
        default:
            CV_Error(Error::StsBadArg, "bytes must be 16, 32, or 64");
    }
}

int BriefDescriptorExtractorImpl::descriptorSize() const
{
    return bytes_;
}

int BriefDescriptorExtractorImpl::descriptorType() const
{
    return CV_8UC1;
}

int BriefDescriptorExtractorImpl::defaultNorm() const
{
    return NORM_HAMMING;
}

void BriefDescriptorExtractorImpl::read( const FileNode& fn)
{
    int dSize = fn["descriptorSize"];
    switch (dSize)
    {
        case 16:
            test_fn_ = pixelTests16;
            break;
        case 32:
            test_fn_ = pixelTests32;
            break;
        case 64:
            test_fn_ = pixelTests64;
            break;
        default:
            CV_Error(Error::StsBadArg, "descriptorSize must be 16, 32, or 64");
    }
    bytes_ = dSize;
}

void BriefDescriptorExtractorImpl::write( FileStorage& fs) const
{
    fs << "descriptorSize" << bytes_;
}

void BriefDescriptorExtractorImpl::compute(InputArray image,
                                           std::vector<KeyPoint>& keypoints,
                                           OutputArray descriptors)
{
    // Construct integral image for fast smoothing (box filter)
    Mat sum;

    Mat grayImage = image.getMat();
    if( image.type() != CV_8U ) cvtColor( image, grayImage, COLOR_BGR2GRAY );

    ///TODO allow the user to pass in a precomputed integral image
    //if(image.type() == CV_32S)
    //  sum = image;
    //else

    integral( grayImage, sum, CV_32S);

    //Remove keypoints very close to the border
    KeyPointsFilter::runByImageBorder(keypoints, image.size(), PATCH_SIZE/2 + KERNEL_SIZE/2);

    descriptors.create((int)keypoints.size(), bytes_, CV_8U);
    descriptors.setTo(Scalar::all(0));
    test_fn_(sum, keypoints, descriptors, use_orientation_);
}

}
} // namespace cv
