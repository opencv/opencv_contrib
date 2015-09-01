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
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2\calib3d.hpp>

namespace cv
{
namespace optflow
{

    class SiftImageExtractorImpl : public SiftImageExtractor
    {
    public:
        //!	Default Constructor.
        SiftImageExtractorImpl(int scale_format = SiftImageExtractor::SCALE_UNIFORM);

        void compute(const Mat& img, Mat& siftImg);

        void compute(const Mat& img0, const Mat& img1, Mat& siftImg0, Mat& siftImg1);

        void compute(const Mat& img0, const Mat& img1,
            const std::vector<KeyPoint>& keypoints0, const std::vector<KeyPoint>& keypoints1,
            Mat& siftImg0, Mat& siftImg1);

    private:
        void compute(const Mat& img, Mat& siftImg, const Mat& scalemap);

        void findSparseMatches(const Mat& img0, const Mat& img1,
            std::vector<KeyPoint>& keypoints0, std::vector<KeyPoint>& keypoints1);

        void symmetryFilter(std::vector<DMatch>& matches1, std::vector<DMatch>& matches2,
            std::vector<DMatch>& symMatches);

        void geometricFilter(const std::vector<KeyPoint>& keypoints1,
            const std::vector<KeyPoint>& keypoints2,
            const std::vector<DMatch>& inliers, std::vector<DMatch>& geoMatches);

    private:
        Ptr<ScaleMap> mScaleMap;
    };

    Ptr<SiftImageExtractor> SiftImageExtractor::create(int scale_format)
    {
        return makePtr<SiftImageExtractorImpl>(scale_format);
    }

    SiftImageExtractorImpl::SiftImageExtractorImpl(int scale_format)
    {
        if (scale_format == SCALE_UNIFORM) return;

        mScaleMap = ScaleMap::create(scale_format == SCALE_EXP);
    }

    void SiftImageExtractorImpl::compute(const Mat& img, Mat& siftImg)
    {
        Mat_<float> scalemap(img.rows, img.cols, 3.0f);
        compute(img, siftImg, scalemap);
    }

    void SiftImageExtractorImpl::compute(const Mat& img0, const Mat& img1,
        Mat& siftImg0, Mat& siftImg1)
    {
        std::vector<KeyPoint> keypoints0, keypoints1;
        findSparseMatches(img0, img1, keypoints0, keypoints1);
        compute(img0, img1, keypoints0, keypoints1, siftImg0, siftImg1);
    }

    void SiftImageExtractorImpl::compute(const Mat& img0, const Mat& img1,
        const std::vector<KeyPoint>& keypoints0, const std::vector<KeyPoint>& keypoints1,
        Mat& siftImg0, Mat& siftImg1)
    {
        // Compute scale maps
        Mat scalemap0, scalemap1;
        mScaleMap->compute(img0, keypoints0, scalemap0);
        mScaleMap->compute(img1, keypoints1, scalemap1);

        // Compute SIFT images
        compute(img0, siftImg0, scalemap0);
        compute(img1, siftImg1, scalemap1);
    }

    void SiftImageExtractorImpl::compute(const Mat& img, Mat& siftImg, const Mat& scalemap)
    {
        cv::Ptr<xfeatures2d::SIFT> siftExtractor = xfeatures2d::SIFT::create();
        int r, c, index = 0;

        std::vector<cv::KeyPoint> keypoints(img.rows*img.cols);

        if (scalemap.empty())
        {
            for (r = 0; r < img.rows; r += 1)
            {
                for (c = 0; c < img.cols; c += 1)
                {
                    cv::KeyPoint& key = keypoints[index++];
                    key.pt.x = c;
                    key.pt.y = r;
                    key.size = 3.0f;
                }
            }
        }
        else
        {
            float* scalemap_data = (float*)scalemap.data;
            for (r = 0; r < img.rows; r += 1)
            {
                for (c = 0; c < img.cols; c += 1)
                {
                    cv::KeyPoint& key = keypoints[index++];
                    key.pt.x = c;
                    key.pt.y = r;
                    key.size = *scalemap_data++;
                }
            }
        }

        cv::Mat desc;

        cv::Mat bimg;
        img.convertTo(bimg, CV_8U);
        siftExtractor->compute(bimg, keypoints, desc);

        // Output sift image
        siftImg.create(img.rows, img.cols, CV_32FC(128));
        memcpy(siftImg.data, desc.data, desc.total()*sizeof(float));
    }

    void SiftImageExtractorImpl::findSparseMatches(const Mat& img0, const Mat& img1,
        std::vector<KeyPoint>& keypoints0, std::vector<KeyPoint>& keypoints1)
    {
        Ptr<Feature2D> siftFeatures = xfeatures2d::SIFT::create();

        // Step 1: Detect the keypoints using SIFT Detector
        std::vector<KeyPoint> rawKeypoints0, rawKeypoints1;
        siftFeatures->detect(img0, rawKeypoints0);
        siftFeatures->detect(img1, rawKeypoints1);

        // Step 2: Calculate descriptors
        Mat descriptors0, descriptors1;
        siftFeatures->compute(img0, rawKeypoints0, descriptors0);
        siftFeatures->compute(img1, rawKeypoints1, descriptors1);

        // Step 3: Match descriptors using FLANN matcher from both sides
        FlannBasedMatcher matcher;
        std::vector<DMatch> matches0, matches1;
        matcher.match(descriptors0, descriptors1, matches0);
        matcher.match(descriptors1, descriptors0, matches1);

        // Step 4: Filter non symmetric matches
        std::vector<DMatch> symMatches;
        symmetryFilter(matches0, matches1, symMatches);

        // Step 5: Do geometric filtering
        std::vector<DMatch> geoMatches;
        geometricFilter(rawKeypoints0, rawKeypoints1, symMatches, geoMatches);

        // Step 6: Return matching keypoint pairs
        keypoints0.resize(geoMatches.size());
        keypoints1.resize(geoMatches.size());
        for (size_t i = 0; i < geoMatches.size(); ++i)
        {
            keypoints0[i] = rawKeypoints0[geoMatches[i].queryIdx];
            keypoints1[i] = rawKeypoints1[geoMatches[i].trainIdx];
        }
    }

    void SiftImageExtractorImpl::symmetryFilter(std::vector<DMatch>& matches1,
        std::vector<DMatch>& matches2, std::vector<DMatch>& symMatches)
    {
        // query refer to the first and train to the second

        symMatches.reserve(matches1.size());

        // Sort the first matches by train index
        std::sort(matches2.begin(), matches2.end(), [](DMatch& m1, DMatch& m2) {
            return m1.trainIdx < m2.trainIdx;
        });

        // Find symmetric matches
        size_t i = 0, j = 0;
        while (i < matches1.size() && j < matches2.size())
        {
            if (matches1[i].queryIdx < matches2[j].trainIdx) ++i;
            else if (matches1[i].queryIdx > matches2[j].trainIdx) ++j;
            else if (matches1[i].trainIdx == matches2[j++].queryIdx)
                symMatches.push_back(matches1[i]);
        }
    }

    void SiftImageExtractorImpl::geometricFilter(const std::vector<KeyPoint>& keypoints1,
        const std::vector<KeyPoint>& keypoints2,
        const std::vector<DMatch>& inliers, std::vector<DMatch>& geoMatches)
    {
        // Remove previous outliers and convert to points
        std::vector<Point2f> points1(inliers.size()), points2(inliers.size());
        for (size_t i = 0; i < inliers.size(); ++i)
        {
            points1[i] = keypoints1[inliers[i].queryIdx].pt;
            points2[i] = keypoints2[inliers[i].trainIdx].pt;
        }

        // RANSAC with Fundamental matrix model
        std::vector<unsigned char> mask(points1.size());
        Mat F = findFundamentalMat(points1, points2, FM_RANSAC, 3.0, 0.99, mask);

        // Calculate matches without geoemtric outliers
        geoMatches.reserve(mask.size());
        for (size_t i = 0; i < mask.size(); ++i)
        {
            if (!mask[i]) continue;

            geoMatches.push_back(inliers[i]);
        }
    }

}//optflow
}//cv
