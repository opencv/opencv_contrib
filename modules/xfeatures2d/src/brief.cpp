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

    void read( const FileNode& ) CV_OVERRIDE;
    void write( FileStorage& ) const CV_OVERRIDE;

    int descriptorSize() const CV_OVERRIDE;
    int descriptorType() const CV_OVERRIDE;
    int defaultNorm() const CV_OVERRIDE;

    void setDescriptorSize(int bytes) CV_OVERRIDE;
    int getDescriptorSize() const CV_OVERRIDE { return bytes_;}

    void setUseOrientation(bool use_orientation) CV_OVERRIDE { use_orientation_ = use_orientation; }
    bool getUseOrientation() const CV_OVERRIDE { return use_orientation_; };

    void compute(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors) CV_OVERRIDE;

protected:
    typedef void(*PixelTestFn)(InputArray, const std::vector<KeyPoint>&, OutputArray, bool use_orientation );

    int bytes_;
    bool use_orientation_;
    PixelTestFn test_fn_;
};

Ptr<BriefDescriptorExtractor> BriefDescriptorExtractor::create( int bytes, bool use_orientation )
{
    return makePtr<BriefDescriptorExtractorImpl>(bytes, use_orientation );
}

String BriefDescriptorExtractor::getDefaultName() const
{
    return (Feature2D::getDefaultName() + ".BRIEF");
}

inline int smoothedSum(const Mat& sum, const KeyPoint& pt, int y, int x, bool use_orientation, Matx21f R)
{
    static const int HALF_KERNEL = BriefDescriptorExtractorImpl::KERNEL_SIZE / 2;

    if ( use_orientation )
    {
      int rx = (int)(((float)x)*R(1,0) - ((float)y)*R(0,0));
      int ry = (int)(((float)x)*R(0,0) + ((float)y)*R(1,0));
      if (rx > 24) rx = 24;
      if (rx < -24) rx = -24;
      if (ry > 24) ry = 24;
      if (ry < -24) ry = -24;
      x = rx; y = ry;
    }
    const int img_y = (int)(pt.pt.y + 0.5) + y;
    const int img_x = (int)(pt.pt.x + 0.5) + x;
    return   sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1)
           - sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL)
           - sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1)
           + sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL);
}

static void pixelTests16(InputArray _sum, const std::vector<KeyPoint>& keypoints, OutputArray _descriptors, bool use_orientation )
{
    Matx21f R;
    Mat sum = _sum.getMat(), descriptors = _descriptors.getMat();
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        uchar* desc = descriptors.ptr(static_cast<int>(i));
        const KeyPoint& pt = keypoints[i];
        if ( use_orientation )
        {
          float angle = pt.angle;
          angle *= (float)(CV_PI/180.f);
          R(0,0) = sin(angle);
          R(1,0) = cos(angle);
        }

#include "generated_16.i"
    }
}

static void pixelTests32(InputArray _sum, const std::vector<KeyPoint>& keypoints, OutputArray _descriptors, bool use_orientation)
{
    Matx21f R;
    Mat sum = _sum.getMat(), descriptors = _descriptors.getMat();
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        uchar* desc = descriptors.ptr(static_cast<int>(i));
        const KeyPoint& pt = keypoints[i];
        if ( use_orientation )
        {
          float angle = pt.angle;
          angle *= (float)(CV_PI / 180.f);
          R(0,0) = sin(angle);
          R(1,0) = cos(angle);
        }

#include "generated_32.i"
    }
}

static void pixelTests64(InputArray _sum, const std::vector<KeyPoint>& keypoints, OutputArray _descriptors, bool use_orientation)
{
    Matx21f R;
    Mat sum = _sum.getMat(), descriptors = _descriptors.getMat();
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        uchar* desc = descriptors.ptr(static_cast<int>(i));
        const KeyPoint& pt = keypoints[i];
        if ( use_orientation )
        {
          float angle = pt.angle;
          angle *= (float)(CV_PI/180.f);
          R(0,0) = sin(angle);
          R(1,0) = cos(angle);
        }

#include "generated_64.i"
    }
}

BriefDescriptorExtractorImpl::BriefDescriptorExtractorImpl(int bytes, bool use_orientation) :
    bytes_(bytes), use_orientation_(use_orientation), test_fn_(NULL)
{
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

void BriefDescriptorExtractorImpl::setDescriptorSize(int bytes)
{
    bytes_ = bytes;
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
    // if node is empty, keep previous value
    if (!fn["descriptorSize"].empty())
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
    if (!fn["use_orientation"].empty())
        fn["use_orientation"] >> use_orientation_;
}

void BriefDescriptorExtractorImpl::write( FileStorage& fs) const
{
    if ( fs.isOpened() )
    {
        fs << "name" << getDefaultName();
        fs << "descriptorSize" << bytes_;
        fs << "use_orientation" << use_orientation_;
    }
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
