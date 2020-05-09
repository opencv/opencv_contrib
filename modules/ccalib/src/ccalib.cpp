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
 // Copyright (C) 2014, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_CCALIB_CPP__
#define __OPENCV_CCALIB_CPP__
#ifdef __cplusplus

#include "precomp.hpp"
#include "opencv2/ccalib.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h> // CV_TERM
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include <vector>
#include <cstring>

namespace cv{ namespace ccalib{

using namespace std;

const int MIN_CONTOUR_AREA_PX      = 100;
const float MIN_CONTOUR_AREA_RATIO = 0.2f;
const float MAX_CONTOUR_AREA_RATIO = 5.0f;

const int MIN_POINTS_FOR_H         = 10;

const float MAX_PROJ_ERROR_PX      = 5.0f;

CustomPattern::CustomPattern()
{
    initialized = false;
}

bool CustomPattern::create(InputArray pattern, const Size2f boardSize, OutputArray output)
{
    CV_Assert(!pattern.empty() && (boardSize.area() > 0));

    Mat img = pattern.getMat();
    float pixel_size = (boardSize.width > boardSize.height)?    // Choose the longer side for more accurate calculation
                         float(img.cols) / boardSize.width:     // width is longer
                         float(img.rows) / boardSize.height;    // height is longer
    return init(img, pixel_size, output);
}

bool CustomPattern::init(Mat& image, const float pixel_size, OutputArray output)
{
    image.copyTo(img_roi);
    //Setup object corners
    obj_corners = std::vector<Point2f>(4);
    obj_corners[0] = Point2f(0, 0); obj_corners[1] = Point2f(float(img_roi.cols), 0);
    obj_corners[2] = Point2f(float(img_roi.cols), float(img_roi.rows)); obj_corners[3] = Point2f(0, float(img_roi.rows));

    if (!detector)   // if no detector chosen, use default
    {
        Ptr<ORB> orb = ORB::create();
        orb->setMaxFeatures(2000);
        orb->setScaleFactor(1.15);
        orb->setNLevels(30);
        detector = orb;
    }

    detector->detect(img_roi, keypoints);
    if (keypoints.empty())
    {
        initialized = false;
        return initialized;
    }
    refineKeypointsPos(img_roi, keypoints);

    if (!descriptorExtractor)   // if no extractor chosen, use default
        descriptorExtractor = ORB::create();
    descriptorExtractor->compute(img_roi, keypoints, descriptor);

    if (!descriptorMatcher)
        descriptorMatcher = DescriptorMatcher::create("BruteForce-Hamming(2)");

    // Scale found points by pixelSize
    pxSize = pixel_size;
    scaleFoundPoints(pxSize, keypoints, points3d);

    if (output.needed())
    {
        Mat out;
        drawKeypoints(img_roi, keypoints, out, Scalar(0, 0, 255));
        out.copyTo(output);
    }

    initialized = !keypoints.empty();
    return initialized; // initialized if any keypoints are found
}

CustomPattern::~CustomPattern() {}

bool CustomPattern::isInitialized()
{
    return initialized;
}

bool CustomPattern::setFeatureDetector(Ptr<FeatureDetector> featureDetector)
{
    if (!initialized)
    {
        this->detector = featureDetector;
        return true;
    }
    else
        return false;
}

bool CustomPattern::setDescriptorExtractor(Ptr<DescriptorExtractor> extractor)
{
    if (!initialized)
    {
        this->descriptorExtractor = extractor;
        return true;
    }
    else
        return false;
}

bool CustomPattern::setDescriptorMatcher(Ptr<DescriptorMatcher> matcher)
{
    if (!initialized)
    {
        this->descriptorMatcher = matcher;
        return true;
    }
    else
        return false;
}

Ptr<FeatureDetector> CustomPattern::getFeatureDetector()
{
    return detector;
}

Ptr<DescriptorExtractor> CustomPattern::getDescriptorExtractor()
{
    return descriptorExtractor;
}

Ptr<DescriptorMatcher> CustomPattern::getDescriptorMatcher()
{
    return descriptorMatcher;
}

void CustomPattern::scaleFoundPoints(const double pixelSize,
            const vector<KeyPoint>& corners, vector<Point3f>& pts3d)
{
    for (unsigned int i = 0; i < corners.size(); ++i)
    {
        pts3d.push_back(Point3f(
                float(corners[i].pt.x * pixelSize),
                float(corners[i].pt.y * pixelSize),
                0));
    }
}

//Takes a descriptor and turns it into an (x,y) point
void CustomPattern::keypoints2points(const vector<KeyPoint>& in, vector<Point2f>& out)
{
    out.clear();
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        out.push_back(in[i].pt);
    }
}

void CustomPattern::updateKeypointsPos(vector<KeyPoint>& in, const vector<Point2f>& new_pos)
{
    for (size_t i = 0; i < in.size(); ++i)
    {
        in[i].pt= new_pos[i];
    }
}

void CustomPattern::refinePointsPos(const Mat& img, vector<Point2f>& p)
{
    Mat gray;
    cvtColor(img, gray, COLOR_RGB2GRAY);
    cornerSubPix(gray, p, Size(10, 10), Size(-1, -1),
                TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 0.1));

}

void CustomPattern::refineKeypointsPos(const Mat& img, vector<KeyPoint>& kp)
{
    vector<Point2f> points;
    keypoints2points(kp, points);
    refinePointsPos(img, points);
    updateKeypointsPos(kp, points);
}

template<typename Tstve>
void deleteStdVecElem(vector<Tstve>& v, int idx)
{
    v[idx] = v.back();
    v.pop_back();
}

void CustomPattern::check_matches(vector<Point2f>& matched, const vector<Point2f>& pattern, vector<DMatch>& good,
                                  vector<Point3f>& pattern_3d, const Mat& H)
{
    vector<Point2f> proj;
    perspectiveTransform(pattern, proj, H);

    int deleted = 0;
    double error_sum = 0;
    double error_sum_filtered = 0;
    for (uint i = 0; i < proj.size(); ++i)
    {
        double error = norm(matched[i] - proj[i]);
        error_sum += error;
        if (error >= MAX_PROJ_ERROR_PX)
        {
            deleteStdVecElem(good, i);
            deleteStdVecElem(matched, i);
            deleteStdVecElem(pattern_3d, i);
            ++deleted;
        }
        else
        {
            error_sum_filtered += error;
        }
    }
}

bool CustomPattern::findPatternPass(const Mat& image, vector<Point2f>& matched_features, vector<Point3f>& pattern_points,
                                    Mat& H, vector<Point2f>& scene_corners, const double pratio, const double proj_error,
                                    const bool refine_position, const Mat& mask, OutputArray output)
{
    if (!initialized) {return false; }
    matched_features.clear();
    pattern_points.clear();

    vector<vector<DMatch> > matches;
    vector<KeyPoint> f_keypoints;
    Mat f_descriptor;

    detector->detect(image, f_keypoints, mask);
    if (refine_position) refineKeypointsPos(image, f_keypoints);

    descriptorExtractor->compute(image, f_keypoints, f_descriptor);
    descriptorMatcher->knnMatch(f_descriptor, descriptor, matches, 2); // k = 2;
    vector<DMatch> good_matches;
    vector<Point2f> obj_points;

    for(int i = 0; i < f_descriptor.rows; ++i)
    {
        if(matches[i][0].distance < pratio * matches[i][1].distance)
        {
            const DMatch& dm = matches[i][0];
            good_matches.push_back(dm);
            // "keypoints1[matches[i].queryIdx] has a corresponding point in keypoints2[matches[i].trainIdx]"
            matched_features.push_back(f_keypoints[dm.queryIdx].pt);
            pattern_points.push_back(points3d[dm.trainIdx]);
            obj_points.push_back(keypoints[dm.trainIdx].pt);
        }
    }

    if (good_matches.size() < MIN_POINTS_FOR_H) return false;

    Mat h_mask;
    H = findHomography(obj_points, matched_features, RANSAC, proj_error, h_mask);
    if (H.empty())
    {
        // cout << "findHomography() returned empty Mat." << endl;
        return false;
    }

    for(unsigned int i = 0; i < good_matches.size(); ++i)
    {
        if(!h_mask.data[i])
        {
            deleteStdVecElem(good_matches, i);
            deleteStdVecElem(matched_features, i);
            deleteStdVecElem(pattern_points, i);
        }
    }

    if (good_matches.empty()) return false;

    size_t numb_elem = good_matches.size();
    check_matches(matched_features, obj_points, good_matches, pattern_points, H);
    if (good_matches.empty() || numb_elem < good_matches.size()) return false;

    // Get the corners from the image
    scene_corners = vector<Point2f>(4);
    perspectiveTransform(obj_corners, scene_corners, H);

    // Check correctnes of H
    // Is it a convex hull?
    bool cConvex = isContourConvex(scene_corners);
    if (!cConvex) return false;

    // Is the hull too large or small?
    double scene_area = contourArea(scene_corners);
    if (scene_area < MIN_CONTOUR_AREA_PX) return false;
    double ratio = scene_area/img_roi.size().area();
    if ((ratio < MIN_CONTOUR_AREA_RATIO) ||
        (ratio > MAX_CONTOUR_AREA_RATIO)) return false;

    // Is any of the projected points outside the hull?
    for(unsigned int i = 0; i < good_matches.size(); ++i)
    {
        if(pointPolygonTest(scene_corners, f_keypoints[good_matches[i].queryIdx].pt, false) < 0)
        {
            deleteStdVecElem(good_matches, i);
            deleteStdVecElem(matched_features, i);
            deleteStdVecElem(pattern_points, i);
        }
    }

    if (output.needed())
    {
        Mat out;
        drawMatches(image, f_keypoints, img_roi, keypoints, good_matches, out);
        // Draw lines between the corners (the mapped object in the scene - image_2 )
        line(out, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 2);
        line(out, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 2);
        line(out, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 2);
        line(out, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 2);
        out.copyTo(output);
    }

    return (!good_matches.empty()); // return true if there are enough good matches
}

bool CustomPattern::findPattern(InputArray image, OutputArray matched_features, OutputArray pattern_points,
                                const double ratio, const double proj_error, const bool refine_position, OutputArray out,
                                OutputArray H, OutputArray pattern_corners)
{
    CV_Assert(!image.empty() && proj_error > 0);

    Mat img = image.getMat();
    vector<Point2f> m_ftrs;
    vector<Point3f> pattern_pts;
    Mat _H;
    vector<Point2f> scene_corners;
    if (!findPatternPass(img, m_ftrs, pattern_pts, _H, scene_corners, 0.6, proj_error, refine_position))
        return false; // pattern not found

    Mat mask = Mat::zeros(img.size(), CV_8UC1);
    vector<vector<Point> > obj(1);
    vector<Point> scorners_int(scene_corners.size());
    for (uint i = 0; i < scene_corners.size(); ++i)
        scorners_int[i] = (Point)scene_corners[i]; // for drawContours
    obj[0] = scorners_int;
    drawContours(mask, obj, 0, Scalar(255), FILLED);

    // Second pass
    Mat output;
    if (!findPatternPass(img, m_ftrs, pattern_pts, _H, scene_corners,
                         ratio, proj_error, refine_position, mask, output))
        return false; // pattern not found

    Mat(m_ftrs).copyTo(matched_features);
    Mat(pattern_pts).copyTo(pattern_points);
    if (out.needed()) output.copyTo(out);
    if (H.needed()) _H.copyTo(H);
    if (pattern_corners.needed()) Mat(scene_corners).copyTo(pattern_corners);

    return (!m_ftrs.empty());
}

void CustomPattern::getPatternPoints(std::vector<KeyPoint>& original_points)
{
    original_points = keypoints;
}

double CustomPattern::getPixelSize()
{
    return pxSize;
}

double CustomPattern::calibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints,
                Size imageSize, InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags,
                TermCriteria criteria)
{
    return calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs,
                            rvecs, tvecs, flags, criteria);
}

bool CustomPattern::findRt(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix,
                           InputArray distCoeffs, InputOutputArray rvec, InputOutputArray tvec, bool useExtrinsicGuess, int flags)
{
    return solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, flags);
}

bool CustomPattern::findRt(InputArray image, InputArray cameraMatrix, InputArray distCoeffs,
                           InputOutputArray rvec, InputOutputArray tvec, bool useExtrinsicGuess, int flags)
{
    vector<Point2f> imagePoints;
    vector<Point3f> objectPoints;

    if (!findPattern(image, imagePoints, objectPoints))
        return false;
    return solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, flags);
}

bool CustomPattern::findRtRANSAC(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs,
                                 InputOutputArray rvec, InputOutputArray tvec, bool useExtrinsicGuess, int iterationsCount,
                                 float reprojectionError, int minInliersCount, OutputArray inliers, int flags)
{
    int npoints = imagePoints.getMat().checkVector(2);
    CV_Assert(npoints > 0);
    double confidence_factor = (double)minInliersCount / (double)npoints;
    double confidence = confidence_factor < 0.001 ? 0.001 : confidence_factor > 0.999 ? 0.999 : confidence_factor;

    solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess,
                   iterationsCount, reprojectionError, confidence, inliers, flags);
    return true; // for consistency with the other methods
}

bool CustomPattern::findRtRANSAC(InputArray image, InputArray cameraMatrix, InputArray distCoeffs,
                                 InputOutputArray rvec, InputOutputArray tvec, bool useExtrinsicGuess, int iterationsCount,
                                 float reprojectionError, int minInliersCount, OutputArray inliers, int flags)
{
    vector<Point2f> imagePoints;
    vector<Point3f> objectPoints;

    if (!findPattern(image, imagePoints, objectPoints))
        return false;

    double confidence_factor = (double)minInliersCount / (double)imagePoints.size();
    double confidence = confidence_factor < 0.001 ? 0.001 : confidence_factor > 0.999 ? 0.999 : confidence_factor;

    solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess,
                   iterationsCount, reprojectionError, confidence, inliers, flags);
    return true;
}

void CustomPattern::drawOrientation(InputOutputArray image, InputArray tvec, InputArray rvec,
                                    InputArray cameraMatrix, InputArray distCoeffs,
                                    double axis_length, int axis_width)
{
    Point3f ptrCtr3d = Point3f(float((img_roi.cols * pxSize)/2.0), float((img_roi.rows * pxSize)/2.0), 0);

    vector<Point3f> axis(4);
    float alen = float(axis_length * pxSize);
    axis[0] = ptrCtr3d;
    axis[1] = Point3f(alen, 0, 0) + ptrCtr3d;
    axis[2] = Point3f(0, alen, 0) + ptrCtr3d;
    axis[3] = Point3f(0, 0, -alen) + ptrCtr3d;

    vector<Point2f> proj_axis;
    projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs, proj_axis);

    Mat img = image.getMat();
    line(img, proj_axis[0], proj_axis[1], Scalar(0, 0, 255), axis_width); // red
    line(img, proj_axis[0], proj_axis[2], Scalar(0, 255, 0), axis_width); // green
    line(img, proj_axis[0], proj_axis[3], Scalar(255, 0, 0), axis_width); // blue

    img.copyTo(image);
}

}} // namespace ccalib, cv

#endif // __OPENCV_CCALIB_CPP__
#endif // cplusplus

