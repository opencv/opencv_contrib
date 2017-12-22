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
// Copyright (C) 2015, Baisheng Lai (laibaisheng@gmail.com), Zhejiang University,
// all rights reserved.
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

/**
 * This module was accepted as a GSoC 2015 project for OpenCV, authored by
 * Baisheng Lai, mentored by Bo Li.

 * This module implements points extraction from a "random" pattern image.
 * It returns 3D object points and 2D image pixels that can be used for calibration.
 * Compared with traditional calibration object like chessboard, it is useful when pattern is
 * partly occluded or only a part of pattern can be observed in multiple cameras calibration.
 * For detail, refer to this paper
 * B. Li, L. Heng, K. Kevin  and M. Pollefeys, "A Multiple-Camera System
 * Calibration Toolbox Using A Feature Descriptor-Based Calibration
 * Pattern", in IROS 2013.
 */
#include "precomp.hpp"
#include "opencv2/ccalib/randpattern.hpp"
#include <iostream>
using namespace std;

namespace cv { namespace randpattern {
RandomPatternCornerFinder::RandomPatternCornerFinder(float patternWidth, float patternHeight,
    int nminiMatch, int depth, int verbose, int showExtraction, Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> descriptor,
    Ptr<DescriptorMatcher> matcher)
{
    _patternHeight = patternHeight;
    _patternWidth = patternWidth;
    _nminiMatch = nminiMatch;
    _objectPonits.resize(0);
    _imagePoints.resize(0);
    _depth = depth;
    _detector = detector;
    _descriptor = descriptor;
    _matcher = matcher;
    _showExtraction = showExtraction;
	_verbose = verbose;
}

void RandomPatternCornerFinder::computeObjectImagePoints(std::vector<cv::Mat> inputImages)
{
    CV_Assert(!_patternImage.empty());
    CV_Assert(inputImages.size() > 0);

    int nImages = (int)inputImages.size();
    std::vector<Mat> imageObjectPoints;
    for (int i = 0; i < nImages; ++i)
    {
        imageObjectPoints = computeObjectImagePointsForSingle(inputImages[i]);
        if ((int)imageObjectPoints[0].total() > _nminiMatch)
        {
            _imagePoints.push_back(imageObjectPoints[0]);
            _objectPonits.push_back(imageObjectPoints[1]);
        }
    }
}

void RandomPatternCornerFinder::keyPoints2MatchedLocation(const std::vector<cv::KeyPoint>& imageKeypoints,
    const std::vector<cv::KeyPoint>& patternKeypoints, const std::vector<cv::DMatch> matchces,
    cv::Mat& matchedImagelocation, cv::Mat& matchedPatternLocation)
{
    matchedImagelocation.release();
    matchedPatternLocation.release();
    std::vector<Vec2d> image, pattern;
    for (int i = 0; i < (int)matchces.size(); ++i)
    {
        Point2f imgPt = imageKeypoints[matchces[i].queryIdx].pt;
        Point2f patPt = patternKeypoints[matchces[i].trainIdx].pt;
        image.push_back(Vec2d(imgPt.x, imgPt.y));
        pattern.push_back(Vec2d(patPt.x, patPt.y));
    }
    Mat(image).convertTo(matchedImagelocation, CV_64FC2);
    Mat(pattern).convertTo(matchedPatternLocation, CV_64FC2);
}

void RandomPatternCornerFinder::getFilteredLocation(cv::Mat& imageKeypoints, cv::Mat& patternKeypoints, const cv::Mat mask)
{
    Mat tmpKeypoint, tmpPattern;
    imageKeypoints.copyTo(tmpKeypoint);
    patternKeypoints.copyTo(tmpPattern);
    imageKeypoints.release();
    patternKeypoints.release();
    std::vector<cv::Vec2d> vecKeypoint, vecPattern;
    for(int i = 0; i < (int)mask.total(); ++i)
    {
        if (mask.at<uchar>(i) == 1)
        {
            vecKeypoint.push_back(tmpKeypoint.at<Vec2d>(i));
            vecPattern.push_back(tmpPattern.at<Vec2d>(i));
        }
    }
    Mat(vecKeypoint).convertTo(imageKeypoints, CV_64FC2);
    Mat(vecPattern).convertTo(patternKeypoints, CV_64FC2);
}

void RandomPatternCornerFinder::getObjectImagePoints(const cv::Mat& imageKeypoints, const cv::Mat& patternKeypoints)
{
    Mat imagePoints_i, objectPoints_i;
    int imagePointsType = CV_MAKETYPE(_depth, 2);
    int objectPointsType = CV_MAKETYPE(_depth, 3);
    imageKeypoints.convertTo(imagePoints_i, imagePointsType);
    _imagePoints.push_back(imagePoints_i);
    if (patternKeypoints.total()!=CV_64FC2)
        patternKeypoints.convertTo(patternKeypoints, CV_64FC2);

    std::vector<Vec3d> objectPoints;

    for (int i = 0; i < (int)patternKeypoints.total(); ++i)
    {
        double x = patternKeypoints.at<Vec2d>(i)[0];
        double y = patternKeypoints.at<Vec2d>(i)[1];
        x = x / _patternImageSize.width * _patternWidth;
        y = y / _patternImageSize.height * _patternHeight;
        objectPoints.push_back(Vec3d(x, y, 0));
    }

    Mat(objectPoints).convertTo(objectPoints_i, objectPointsType);
    _objectPonits.push_back(objectPoints_i);
}

void RandomPatternCornerFinder::crossCheckMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
    const Mat& descriptors1, const Mat& descriptors2,
    std::vector<DMatch>& filteredMatches12, int knn )
{
    filteredMatches12.clear();
    std::vector<std::vector<DMatch> > matches12, matches21;
    descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
    descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );
    for( size_t m = 0; m < matches12.size(); m++ )
    {
        bool findCrossCheck = false;
        for( size_t fk = 0; fk < matches12[m].size(); fk++ )
        {
            DMatch forward = matches12[m][fk];

            for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
            {
                DMatch backward = matches21[forward.trainIdx][bk];
                if( backward.trainIdx == forward.queryIdx )
                {
                    filteredMatches12.push_back(forward);
                    findCrossCheck = true;
                    break;
                }
            }
            if( findCrossCheck ) break;
        }
    }
}

void RandomPatternCornerFinder::drawCorrespondence(const Mat& image1, const std::vector<cv::KeyPoint> keypoint1,
    const Mat& image2, const std::vector<cv::KeyPoint> keypoint2, const std::vector<cv::DMatch> matchces,
    const Mat& mask1, const Mat& mask2, const int step)
{
    Mat img_corr;
    if(step == 1)
    {
        drawMatches(image1, keypoint1, image2, keypoint2, matchces, img_corr);
    }
    else if(step == 2)
    {
        std::vector<cv::DMatch> matchesFilter;
        for (int i = 0; i < (int)mask1.total(); ++i)
        {
            if (!mask1.empty() && mask1.at<uchar>(i) == 1)
            {
                matchesFilter.push_back(matchces[i]);
            }
        }
        drawMatches(image1, keypoint1, image2, keypoint2, matchesFilter, img_corr);
    }
    else if(step == 3)
    {
        std::vector<cv::DMatch> matchesFilter;
        int j = 0;
        for (int i = 0; i < (int)mask1.total(); ++i)
        {
            if (mask1.at<uchar>(i) == 1)
            {
                if (!mask2.empty() && mask2.at<uchar>(j) == 1)
                {
                    matchesFilter.push_back(matchces[i]);
                }
                j++;
            }
        }
        drawMatches(image1, keypoint1, image2, keypoint2, matchesFilter, img_corr);
    }
    imshow("correspondence", img_corr);
    waitKey(0);
}

std::vector<cv::Mat> RandomPatternCornerFinder::getObjectPoints()
{
    return _objectPonits;
}

std::vector<cv::Mat> RandomPatternCornerFinder::getImagePoints()
{
    return _imagePoints;
}

void RandomPatternCornerFinder::loadPattern(cv::Mat patternImage)
{
    _patternImage = patternImage.clone();
    if (_patternImage.type()!= CV_8U)
        _patternImage.convertTo(_patternImage, CV_8U);
    _patternImageSize = _patternImage.size();
    _detector->detect(patternImage, _keypointsPattern);
    _descriptor->compute(patternImage, _keypointsPattern, _descriptorPattern);
    _descriptorPattern.convertTo(_descriptorPattern, CV_32F);
}

std::vector<cv::Mat> RandomPatternCornerFinder::computeObjectImagePointsForSingle(cv::Mat inputImage)
{
    CV_Assert(!_patternImage.empty());
    std::vector<cv::Mat> r(2);
    Mat descriptorImage1, descriptorImage2,descriptorImage;
    std::vector<cv::KeyPoint> keypointsImage1, keypointsImage2, keypointsImage;
    if (inputImage.type()!=CV_8U)
    {
        inputImage.convertTo(inputImage, CV_8U);
    }

    Mat imageEquHist;
    equalizeHist(inputImage, imageEquHist);

    _detector->detect(inputImage, keypointsImage1);
    _descriptor->compute(inputImage, keypointsImage1, descriptorImage1);
    _detector->detect(imageEquHist, keypointsImage2);
    _descriptor->compute(imageEquHist, keypointsImage2, descriptorImage2);
    descriptorImage1.convertTo(descriptorImage1, CV_32F);
    descriptorImage2.convertTo(descriptorImage2, CV_32F);

    // match with pattern
    std::vector<DMatch> matchesImgtoPat, matchesImgtoPat1, matchesImgtoPat2;

    cv::Mat keypointsImageLocation, keypointsPatternLocation;

    crossCheckMatching(this->_matcher, descriptorImage1, this->_descriptorPattern, matchesImgtoPat1, 1);
    crossCheckMatching(this->_matcher, descriptorImage2, this->_descriptorPattern, matchesImgtoPat2, 1);
    if ((int)matchesImgtoPat1.size() > (int)matchesImgtoPat2.size())
    {
        matchesImgtoPat = matchesImgtoPat1;
        keypointsImage = keypointsImage1;
    }
    else
    {
        matchesImgtoPat = matchesImgtoPat2;
        keypointsImage = keypointsImage2;
    }

    keyPoints2MatchedLocation(keypointsImage, this->_keypointsPattern, matchesImgtoPat,
        keypointsImageLocation, keypointsPatternLocation);

    Mat img_corr;

    // innerMask is CV_8U type
    Mat innerMask1, innerMask2;

    // draw raw correspondence
    if(this->_showExtraction)
    {
        drawCorrespondence(inputImage, keypointsImage, _patternImage, _keypointsPattern, matchesImgtoPat,
            innerMask1, innerMask2, 1);
    }

	if (_verbose)
	{
		std::cout << "number of matched points " << (int)keypointsImageLocation.total() << std::endl;
	}
    // outlier remove
    findFundamentalMat(keypointsImageLocation, keypointsPatternLocation,
        FM_RANSAC, 1, 0.995, innerMask1);
    getFilteredLocation(keypointsImageLocation, keypointsPatternLocation, innerMask1);

	if (this->_showExtraction)
	{
		drawCorrespondence(inputImage, keypointsImage, _patternImage, _keypointsPattern, matchesImgtoPat,
			innerMask1, innerMask2, 2);
	}

    findHomography(keypointsImageLocation, keypointsPatternLocation, RANSAC, 30*inputImage.cols/1000, innerMask2);
    getFilteredLocation(keypointsImageLocation, keypointsPatternLocation, innerMask2);

	if (_verbose)
	{
		std::cout << "number of filtered points " << (int)keypointsImageLocation.total() << std::endl;
	}

    // draw filtered correspondence
    if (this->_showExtraction)
    {
        drawCorrespondence(inputImage, keypointsImage, _patternImage, _keypointsPattern, matchesImgtoPat,
            innerMask1, innerMask2, 3);
    }

    std::vector<Vec3d> objectPoints;

    int imagePointsType = CV_MAKETYPE(_depth, 2);
    int objectPointsType = CV_MAKETYPE(_depth, 3);

    keypointsImageLocation.convertTo(r[0], imagePointsType);

    for (int i = 0; i < (int)keypointsPatternLocation.total(); ++i)
    {
        double x = keypointsPatternLocation.at<Vec2d>(i)[0];
        double y = keypointsPatternLocation.at<Vec2d>(i)[1];
        x = x / _patternImageSize.width * _patternWidth;
        y = y / _patternImageSize.height * _patternHeight;
        objectPoints.push_back(Vec3d(x, y, 0));
    }
    Mat(objectPoints).convertTo(r[1], objectPointsType);
    return r;
}

RandomPatternGenerator::RandomPatternGenerator(int imageWidth, int imageHeight)
{
    _imageWidth = imageWidth;
    _imageHeight = imageHeight;
}

void RandomPatternGenerator::generatePattern()
{
    Mat pattern = Mat(_imageHeight, _imageWidth, CV_32F, Scalar(0.));

    int m = 5;
    int count = 0;

    while(m < _imageWidth)
    {
        int n = (int)std::floor(double(_imageHeight) / double(_imageWidth) * double(m)) + 1;

        Mat r = Mat(n, m, CV_32F);
        cv::randn(r, Scalar::all(0), Scalar::all(1));
        cv::resize(r, r, Size(_imageWidth ,_imageHeight));
        double min_r, max_r;
        minMaxLoc(r, &min_r, &max_r);

        r = (r - (float)min_r) / float(max_r - min_r);

        pattern += r;
        count += 1;
        m *= 2;
    }
    pattern = pattern / count * 255;
    pattern.convertTo(pattern, CV_8U);
    equalizeHist(pattern, pattern);
    pattern.copyTo(_pattern);
}

cv::Mat RandomPatternGenerator::getPattern()
{
    return _pattern;
}

}} //namespace randpattern, cv