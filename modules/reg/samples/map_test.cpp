/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
// Copyright (C) 2013, Alfonso Sanchez-Beato, all rights reserved.
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
// In no event shall the contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp> // OpenCV window I/O
#include <opencv2/imgproc.hpp> // OpenCV image transformations

#ifdef COMPARE_FEATURES
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d/calib3d_c.h>
using namespace cv::xfeatures2d;
#endif

#include "opencv2/reg/mapaffine.hpp"
#include "opencv2/reg/mapshift.hpp"
#include "opencv2/reg/mapprojec.hpp"
#include "opencv2/reg/mappergradshift.hpp"
#include "opencv2/reg/mappergradeuclid.hpp"
#include "opencv2/reg/mappergradsimilar.hpp"
#include "opencv2/reg/mappergradaffine.hpp"
#include "opencv2/reg/mappergradproj.hpp"
#include "opencv2/reg/mapperpyramid.hpp"

static const char* DIFF_IM = "Image difference";
static const char* DIFF_REGPIX_IM = "Image difference: pixel registered";

using namespace cv;
using namespace cv::reg;
using namespace std;

static void showDifference(const Mat& image1, const Mat& image2, const char* title)
{
    Mat img1, img2;
    image1.convertTo(img1, CV_32FC3);
    image2.convertTo(img2, CV_32FC3);
    if(img1.channels() != 1)
        cvtColor(img1, img1, COLOR_BGR2GRAY);
    if(img2.channels() != 1)
        cvtColor(img2, img2, COLOR_BGR2GRAY);

    Mat imgDiff;
    img1.copyTo(imgDiff);
    imgDiff -= img2;
    imgDiff /= 2.f;
    imgDiff += 128.f;

    Mat imgSh;
    imgDiff.convertTo(imgSh, CV_8UC3);
    imshow(title, imgSh);
}

static void testShift(const Mat& img1)
{
    Mat img2;

    // Warp original image
    Vec<double, 2> shift(5., 5.);
    MapShift mapTest(shift);
    mapTest.warp(img1, img2);
    showDifference(img1, img2, DIFF_IM);

    // Register
    Ptr<MapperGradShift> mapper = makePtr<MapperGradShift>();
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr = mappPyr.calculate(img1, img2);

    // Print result
    MapShift* mapShift = dynamic_cast<MapShift*>(mapPtr.get());
    cout << endl << "--- Testing shift mapper ---" << endl;
    cout << Mat(shift) << endl;
    cout << Mat(mapShift->getShift()) << endl;

    // Display registration accuracy
    Mat dest;
    mapShift->inverseWarp(img2, dest);
    showDifference(img1, dest, DIFF_REGPIX_IM);

    waitKey(0);
    destroyWindow(DIFF_IM);
    destroyWindow(DIFF_REGPIX_IM);
}

static void testEuclidean(const Mat& img1)
{
    Mat img2;

    // Warp original image
    double theta = 3*CV_PI/180;
    double cosT = cos(theta);
    double sinT = sin(theta);
    Matx<double, 2, 2> linTr(cosT, -sinT, sinT, cosT);
    Vec<double, 2> shift(5., 5.);
    MapAffine mapTest(linTr, shift);
    mapTest.warp(img1, img2);
    showDifference(img1, img2, DIFF_IM);

    // Register
    Ptr<MapperGradEuclid> mapper = makePtr<MapperGradEuclid>();
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr = mappPyr.calculate(img1, img2);

    // Print result
    MapAffine* mapAff = dynamic_cast<MapAffine*>(mapPtr.get());
    cout << endl << "--- Testing Euclidean mapper ---" << endl;
    cout << Mat(linTr) << endl;
    cout << Mat(shift) << endl;
    cout << Mat(mapAff->getLinTr()) << endl;
    cout << Mat(mapAff->getShift()) << endl;

    // Display registration accuracy
    Mat dest;
    mapAff->inverseWarp(img2, dest);
    showDifference(img1, dest, DIFF_REGPIX_IM);

    waitKey(0);
    destroyWindow(DIFF_IM);
    destroyWindow(DIFF_REGPIX_IM);
}

static void testSimilarity(const Mat& img1)
{
    Mat img2;

    // Warp original image
    double theta = 3*CV_PI/180;
    double scale = 0.95;
    double a = scale*cos(theta);
    double b = scale*sin(theta);
    Matx<double, 2, 2> linTr(a, -b, b, a);
    Vec<double, 2> shift(5., 5.);
    MapAffine mapTest(linTr, shift);
    mapTest.warp(img1, img2);
    showDifference(img1, img2, DIFF_IM);

    // Register
    Ptr<MapperGradSimilar> mapper = makePtr<MapperGradSimilar>();
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr = mappPyr.calculate(img1, img2);

    // Print result
    MapAffine* mapAff = dynamic_cast<MapAffine*>(mapPtr.get());
    cout << endl << "--- Testing similarity mapper ---" << endl;
    cout << Mat(linTr) << endl;
    cout << Mat(shift) << endl;
    cout << Mat(mapAff->getLinTr()) << endl;
    cout << Mat(mapAff->getShift()) << endl;

    // Display registration accuracy
    Mat dest;
    mapAff->inverseWarp(img2, dest);
    showDifference(img1, dest, DIFF_REGPIX_IM);

    waitKey(0);
    destroyWindow(DIFF_IM);
    destroyWindow(DIFF_REGPIX_IM);
}

static void testAffine(const Mat& img1)
{
    Mat img2;

    // Warp original image
    Matx<double, 2, 2> linTr(1., 0.1, -0.01, 1.);
    Vec<double, 2> shift(1., 1.);
    MapAffine mapTest(linTr, shift);
    mapTest.warp(img1, img2);
    showDifference(img1, img2, DIFF_IM);

    // Register
    Ptr<MapperGradAffine> mapper = makePtr<MapperGradAffine>();
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr = mappPyr.calculate(img1, img2);

    // Print result
    MapAffine* mapAff = dynamic_cast<MapAffine*>(mapPtr.get());
    cout << endl << "--- Testing affine mapper ---" << endl;
    cout << Mat(linTr) << endl;
    cout << Mat(shift) << endl;
    cout << Mat(mapAff->getLinTr()) << endl;
    cout << Mat(mapAff->getShift()) << endl;

    // Display registration accuracy
    Mat dest;
    mapAff->inverseWarp(img2, dest);
    showDifference(img1, dest, DIFF_REGPIX_IM);

    waitKey(0);
    destroyWindow(DIFF_IM);
    destroyWindow(DIFF_REGPIX_IM);
}

static void testProjective(const Mat& img1)
{
    Mat img2;

    // Warp original image
    Matx<double, 3, 3> projTr(1., 0., 0., 0., 1., 0., 0.0001, 0.0001, 1);
    MapProjec mapTest(projTr);
    mapTest.warp(img1, img2);
    showDifference(img1, img2, DIFF_IM);

    // Register
    Ptr<MapperGradProj> mapper = makePtr<MapperGradProj>();
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr = mappPyr.calculate(img1, img2);

    // Print result
    MapProjec* mapProj = dynamic_cast<MapProjec*>(mapPtr.get());
    mapProj->normalize();
    cout << endl << "--- Testing projective transformation mapper ---" << endl;
    cout << Mat(projTr) << endl;
    cout << Mat(mapProj->getProjTr()) << endl;

    // Display registration accuracy
    Mat dest;
    mapProj->inverseWarp(img2, dest);
    showDifference(img1, dest, DIFF_REGPIX_IM);

    waitKey(0);
    destroyWindow(DIFF_IM);
    destroyWindow(DIFF_REGPIX_IM);
}

#ifdef COMPARE_FEATURES
//
// Following an example from
// http:// ramsrigoutham.com/2012/11/22/panorama-image-stitching-in-opencv/
//
static void calcHomographyFeature(const Mat& image1, const Mat& image2)
{
    static const char* difffeat = "Difference feature registered";

    Mat gray_image1;
    Mat gray_image2;
    // Convert to Grayscale
    if(image1.channels() != 1)
        cvtColor(image1, gray_image1, COLOR_BGR2GRAY);
    else
        image1.copyTo(gray_image1);
    if(image2.channels() != 1)
        cvtColor(image2, gray_image2, COLOR_BGR2GRAY);
    else
        image2.copyTo(gray_image2);

    //-- Step 1: Detect the keypoints using SIFT or SURF Detector
#ifdef USE_SIFT
    Ptr<Feature2D> features = SIFT::create();
#else
    int minHessian = 400;
    Ptr<Feature2D> features = SURF::create(minHessian);
#endif

    std::vector<KeyPoint> keypoints_object, keypoints_scene;

    features->detect(gray_image1, keypoints_object);
    features->detect(gray_image2, keypoints_scene);

    //-- Step 2: Calculate descriptors (feature vectors)

    Mat descriptors_object, descriptors_scene;

    features->compute(gray_image1, keypoints_object, descriptors_object);
    features->compute(gray_image2, keypoints_scene, descriptors_scene);

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(descriptors_object, descriptors_scene, matches);

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for(int i = 0; i < descriptors_object.rows; i++)
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Use only "good" matches (i.e. whose distance is less than 3*min_dist)
    std::vector<DMatch> good_matches;

    for(int i = 0; i < descriptors_object.rows; i++) {
        if(matches[i].distance < 3*min_dist) {
            good_matches.push_back( matches[i]);
        }
    }
    std::vector< Point2f > obj;
    std::vector< Point2f > scene;

    for(size_t i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    // Find the Homography Matrix
    Mat H = findHomography( obj, scene, RANSAC );
    // Use the Homography Matrix to warp the images
    Mat result;
    Mat Hinv = H.inv();
    warpPerspective(image2, result, Hinv, image1.size());

    cout << "--- Feature method\n" << H << endl;

    Mat imf1, resf;
    image1.convertTo(imf1, CV_64FC3);
    result.convertTo(resf, CV_64FC3);
    showDifference(imf1, resf, difffeat);
}

static void calcHomographyPixel(const Mat& img1, const Mat& img2)
{
    static const char* diffpixel = "Difference pixel registered";

    // Register using pixel differences
    Ptr<MapperGradProj> mapper = makePtr<MapperGradProj>();
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr = mappPyr.calculate(img1, img2);

    // Print result
    MapProjec* mapProj = dynamic_cast<MapProjec*>(mapPtr.get());
    mapProj->normalize();
    cout << "--- Pixel-based method\n" << Mat(mapProj->getProjTr()) << endl;

    // Display registration accuracy
    Mat dest;
    mapProj->inverseWarp(img2, dest);
    showDifference(img1, dest, diffpixel);
}

static void comparePixelVsFeature(const Mat& img1_8b, const Mat& img2_8b)
{
    static const char* difforig = "Difference non-registered";

    // Show difference of images
    Mat img1, img2;
    img1_8b.convertTo(img1, CV_64FC3);
    img2_8b.convertTo(img2, CV_64FC3);
    showDifference(img1, img2, difforig);
    cout << endl << "--- Comparing feature-based with pixel difference based ---" << endl;

    // Register using SURF keypoints
    calcHomographyFeature(img1_8b, img2_8b);

    // Register using pixel differences
    calcHomographyPixel(img1, img2);

    waitKey(0);
}
#endif

int main(void)
{
    Mat img1;
    img1 = imread("home.png", IMREAD_UNCHANGED);
    if(!img1.data) {
        cout <<  "Could not open or find file" << endl;
        return -1;
    }
    // Convert to double, 3 channels
    img1.convertTo(img1, CV_64FC3);

    testShift(img1);
    testEuclidean(img1);
    testSimilarity(img1);
    testAffine(img1);
    testProjective(img1);

#ifdef COMPARE_FEATURES
    Mat imgcmp1 = imread("LR_05.png", IMREAD_UNCHANGED);
    if(!imgcmp1.data) {
        cout <<  "Could not open or find file" << endl;
        return -1;
    }

    Mat imgcmp2 = imread("LR_06.png", IMREAD_UNCHANGED);
    if(!imgcmp2.data) {
        cout <<  "Could not open or find file" << endl;
        return -1;
    }
    comparePixelVsFeature(imgcmp1, imgcmp2);
#endif

    return 0;
}
