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
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
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
#include "test_precomp.hpp"

#include <opencv2/structured_light/graycodepattern.hpp>
#include <opencv2/structured_light/sinusoidalpattern.hpp>

namespace opencv_test { namespace {

const string STRUCTURED_LIGHT_DIR = "structured_light";
const string FOLDER_DATA = "data";

TEST( SinusoidalPattern, unwrapPhaseMap )
{
    string folder = cvtest::TS::ptr()->get_data_path() + "/" + STRUCTURED_LIGHT_DIR + "/" + FOLDER_DATA + "/";

    Ptr<structured_light::SinusoidalPattern::Params> paramsPsp, paramsFtp, paramsFaps;
    paramsPsp = makePtr<structured_light::SinusoidalPattern::Params>();
    paramsFtp = makePtr<structured_light::SinusoidalPattern::Params>();
    paramsFaps = makePtr<structured_light::SinusoidalPattern::Params>();
    paramsFtp->methodId = 0;
    paramsPsp->methodId = 1;
    paramsFaps->methodId = 2;

    Ptr<structured_light::SinusoidalPattern> sinusPsp = structured_light::SinusoidalPattern::create(paramsPsp);
    Ptr<structured_light::SinusoidalPattern> sinusFtp = structured_light::SinusoidalPattern::create(paramsFtp);
    Ptr<structured_light::SinusoidalPattern> sinusFaps = structured_light::SinusoidalPattern::create(paramsFaps);

    vector<Mat> captures(3);
    Mat unwrappedPhaseMapPspRef, unwrappedPhaseMapFtpRef, unwrappedPhaseMapFapsRef;
    Mat shadowMask;
    Mat wrappedPhaseMap, unwrappedPhaseMap, unwrappedPhaseMap8;
    captures[0] = imread(folder + "capture_sin_0.jpg", IMREAD_GRAYSCALE);
    captures[1] = imread(folder + "capture_sin_1.jpg", IMREAD_GRAYSCALE);
    captures[2] = imread(folder + "capture_sin_2.jpg", IMREAD_GRAYSCALE);

    unwrappedPhaseMapPspRef = imread(folder + "unwrappedPspTest.jpg", IMREAD_GRAYSCALE);
    unwrappedPhaseMapFtpRef = imread(folder + "unwrappedFtpTest.jpg", IMREAD_GRAYSCALE);
    unwrappedPhaseMapFapsRef = imread(folder + "unwrappedFapsTest.jpg", IMREAD_GRAYSCALE);

    if( !captures[0].data || !captures[1].data || !captures[2].data || !unwrappedPhaseMapFapsRef.data
        || !unwrappedPhaseMapFtpRef.data || !unwrappedPhaseMapPspRef.data )
    {
        cerr << "invalid test data" << endl;
    }

    sinusPsp->computePhaseMap(captures, wrappedPhaseMap, shadowMask);
    sinusPsp->unwrapPhaseMap(wrappedPhaseMap, unwrappedPhaseMap, Size(captures[0].cols, captures[1].rows), shadowMask);
    unwrappedPhaseMap.convertTo(unwrappedPhaseMap8, CV_8U, 1, 128);

    int sumOfDiff = 0;
    int count = 0;
    float ratio = 0;
    for( int i = 0; i < unwrappedPhaseMap8.rows; ++i )
    {
        for( int j = 0; j < unwrappedPhaseMap8.cols; ++j )
        {
            int ref = unwrappedPhaseMapPspRef.at<uchar>(i, j);
            int comp = unwrappedPhaseMap8.at<uchar>(i, j);
            sumOfDiff += (ref - comp);
            count ++;
        }
    }
    ratio = (float)(sumOfDiff / count);

    EXPECT_LE( ratio, 0.003 );

    sinusFtp->computePhaseMap(captures, wrappedPhaseMap, shadowMask);
    sinusFtp->unwrapPhaseMap(wrappedPhaseMap, unwrappedPhaseMap, Size(captures[0].cols, captures[1].rows), shadowMask);
    unwrappedPhaseMap.convertTo(unwrappedPhaseMap8, CV_8U, 1, 128);

    sumOfDiff = 0;
    count = 0;
    ratio = 0;
    for( int i = 0; i < unwrappedPhaseMap8.rows; ++i )
    {
        for( int j = 0; j < unwrappedPhaseMap8.cols; ++j )
        {
            int ref = unwrappedPhaseMapFtpRef.at<uchar>(i, j);
            int comp = unwrappedPhaseMap8.at<uchar>(i, j);
            sumOfDiff += (ref - comp);
            count ++;
        }
    }
    ratio = (float)(sumOfDiff / count);

    EXPECT_LE( ratio, 0.003 );

    sinusFaps->computePhaseMap(captures, wrappedPhaseMap, shadowMask);
    sinusFaps->unwrapPhaseMap(wrappedPhaseMap, unwrappedPhaseMap, Size(captures[0].cols, captures[1].rows), shadowMask);
    unwrappedPhaseMap.convertTo(unwrappedPhaseMap8, CV_8U, 1, 128);

    sumOfDiff = 0;
    count = 0;
    ratio = 0;
    for( int i = 0; i < unwrappedPhaseMap8.rows; ++i )
    {
        for( int j = 0; j < unwrappedPhaseMap8.cols; ++j )
        {
            int ref = unwrappedPhaseMapFapsRef.at<uchar>(i, j);
            int comp = unwrappedPhaseMap8.at<uchar>(i, j);
            sumOfDiff += (ref - comp);
            count ++;
        }
    }
    ratio = (float)(sumOfDiff / count);

    EXPECT_LE( ratio, 0.003 );
}

}} // namespace
