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
    BriefDescriptorExtractorImpl( int bytes = 32, bool use_orientation = false, bool use_scale = false );

    virtual void read( const FileNode& );
    virtual void write( FileStorage& ) const;

    virtual int descriptorSize() const;
    virtual int descriptorType() const;
    virtual int defaultNorm() const;

    virtual void compute(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors);

protected:
    typedef void(*PixelTestFn)(InputArray, const std::vector<KeyPoint>&, OutputArray, bool use_orientation, bool use_scale);

    int bytes_;
    bool use_orientation_;
    bool use_scale_;
    PixelTestFn test_fn_;
};

Ptr<BriefDescriptorExtractor> BriefDescriptorExtractor::create( int bytes, bool use_orientation, bool use_scale )
{
    return makePtr<BriefDescriptorExtractorImpl>(bytes, use_orientation, use_scale);
}

inline int smoothedSum(const Mat& sum, const KeyPoint& pt, int y, int x, bool use_orientation, Matx23f R)
{
    int response;

    static const int HALF_KERNEL = BriefDescriptorExtractorImpl::KERNEL_SIZE / 2;

    // pattern point
    const int img_y = (int)(pt.pt.y + 0.5) + y;
    const int img_x = (int)(pt.pt.x + 0.5) + x;

    // sampling edge
    const int uy = img_y + HALF_KERNEL + 1;
    const int rx = img_x + HALF_KERNEL + 1;
    const int ly = img_y - HALF_KERNEL;
    const int lx = img_x - HALF_KERNEL;

    if ( use_orientation )
    {
        // (x,y)' = R * [x,y]
        const float R00lx = R(0,0)*lx, R10lx = R(1,0)*lx;
        const float R00rx = R(0,0)*rx, R10rx = R(1,0)*rx;
        const float R01ly = R(0,1)*ly, R11ly = R(1,1)*ly;
        const float R01uy = R(0,1)*uy, R11uy = R(1,1)*uy;

        // (uy, rx)
        const int uy0 = (int)(R10rx + R11uy + R(1,2) + 0.5);
        const int rx0 = (int)(R00rx + R01uy + R(0,2) + 0.5);
        // (uy, lx)
        const int uy1 = (int)(R10lx + R11uy + R(1,2) + 0.5);
        const int lx1 = (int)(R00lx + R01uy + R(0,2) + 0.5);
        // (ly, rx)
        const int ly2 = (int)(R10rx + R11ly + R(1,2) + 0.5);
        const int rx2 = (int)(R00rx + R01ly + R(0,2) + 0.5);
        // (ly, lx)
        const int ly3 = (int)(R10lx + R11ly + R(1,2) + 0.5);
        const int lx3 = (int)(R00lx + R01ly + R(0,2) + 0.5);

        // if outside of image
        if ((uy0 > sum.rows) || (uy0 < 0)) return 0;
        if ((rx0 > sum.cols) || (rx0 < 0)) return 0;
        if ((uy1 > sum.rows) || (uy1 < 0)) return 0;
        if ((lx1 > sum.cols) || (lx1 < 0)) return 0;
        if ((ly2 > sum.rows) || (ly2 < 0)) return 0;
        if ((rx2 > sum.cols) || (rx2 < 0)) return 0;
        if ((ly3 > sum.rows) || (ly3 < 0)) return 0;
        if ((lx3 > sum.cols) || (lx3 < 0)) return 0;

        response = sum.at<int>(uy0, rx0)
                 - sum.at<int>(uy1, lx1)
                 - sum.at<int>(ly2, rx2)
                 + sum.at<int>(ly3, lx3);
    }
    else
    {
        response = sum.at<int>(uy, rx)
                 - sum.at<int>(uy, lx)
                 - sum.at<int>(ly, rx)
                 + sum.at<int>(ly, lx);
    }
    return response;
}

static void pixelTests16(InputArray _sum, const std::vector<KeyPoint>& keypoints, OutputArray _descriptors, bool use_orientation, bool use_scale)
{
    Matx23f R;
    Mat sum = _sum.getMat(), descriptors = _descriptors.getMat();
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        uchar* desc = descriptors.ptr(static_cast<int>(i));
        const KeyPoint& pt = keypoints[i];
        if ( use_orientation )
          R = getRotationMatrix2D( pt.pt, -pt.angle, use_scale ? pt.size : 1.0f );

#include "generated_16.i"
    }
}

static void pixelTests32(InputArray _sum, const std::vector<KeyPoint>& keypoints, OutputArray _descriptors, bool use_orientation, bool use_scale)
{
    Matx23f R;
    Mat sum = _sum.getMat(), descriptors = _descriptors.getMat();
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        uchar* desc = descriptors.ptr(static_cast<int>(i));
        const KeyPoint& pt = keypoints[i];
        if ( use_orientation )
          R = getRotationMatrix2D( pt.pt, -pt.angle, use_scale ? pt.size : 1.0f );

#include "generated_32.i"
    }
}

static void pixelTests64(InputArray _sum, const std::vector<KeyPoint>& keypoints, OutputArray _descriptors, bool use_orientation, bool use_scale)
{
    Matx23f R;
    Mat sum = _sum.getMat(), descriptors = _descriptors.getMat();
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        uchar* desc = descriptors.ptr(static_cast<int>(i));
        const KeyPoint& pt = keypoints[i];
        if ( use_orientation )
          R = getRotationMatrix2D( pt.pt, -pt.angle, use_scale ? pt.size : 1.0f );

#include "generated_64.i"
    }
}

BriefDescriptorExtractorImpl::BriefDescriptorExtractorImpl(int bytes, bool use_orientation, bool use_scale) :
    bytes_(bytes), test_fn_(NULL)
{
    use_orientation_ = use_orientation;
    use_scale_ = use_scale;

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
    test_fn_(sum, keypoints, descriptors, use_orientation_, use_scale_);
}

}
} // namespace cv
