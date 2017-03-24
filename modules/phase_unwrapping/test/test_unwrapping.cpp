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

using namespace std;
using namespace cv;

class CV_Unwrapping : public cvtest::BaseTest
{
public:
    CV_Unwrapping();
    ~CV_Unwrapping();
protected:
    void run(int);
};

CV_Unwrapping::CV_Unwrapping(){}

CV_Unwrapping::~CV_Unwrapping(){}

void CV_Unwrapping::run( int )
{
    int rows = 600;
    int cols = 800;
    int max = 50;

    Mat ramp(rows, cols, CV_32FC1);
    Mat wrappedRamp(rows, cols, CV_32FC1);
    Mat unwrappedRamp;
    Mat rowValues(1, cols, CV_32FC1);
    Mat wrappedRowValues(1, cols, CV_32FC1);

    for( int i = 0; i < cols; ++i )
    {
        float v = (float)i*(float)max/(float)cols;
        rowValues.at<float>(0, i) = v;
        wrappedRowValues.at<float>(0, i) = atan2(sin(v), cos(v));
    }
    for( int i = 0; i < rows; ++i )
    {
        rowValues.row(0).copyTo(ramp.row(i));
        wrappedRowValues.row(0).copyTo(wrappedRamp.row(i));
    }

    phase_unwrapping::HistogramPhaseUnwrapping::Params params;
    params.width = cols;
    params.height = rows;
    Ptr<phase_unwrapping::HistogramPhaseUnwrapping> phaseUnwrapping = phase_unwrapping::HistogramPhaseUnwrapping::create(params);
    phaseUnwrapping->unwrapPhaseMap(wrappedRamp, unwrappedRamp);

    for(int i = 0; i < rows; ++i )
    {
        for( int j = 0; j < cols; ++ j )
        {
            EXPECT_NEAR(ramp.at<float>(i, j), unwrappedRamp.at<float>(i, j), 0.001);
        }
    }
}

TEST( HistogramPhaseUnwrapping, unwrapPhaseMap )
{
    CV_Unwrapping test;
    test.safe_run();
}