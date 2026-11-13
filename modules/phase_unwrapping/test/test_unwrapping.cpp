// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

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

TEST( HistogramPhaseUnwrapping, reuse_instance_parity )
{
    const int w = 48, h = 48;
    Mat phase(h, w, CV_32FC1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            phase.at<float>(y, x) = static_cast<float>(std::fmod((x + y) * 0.3, 2 * CV_PI) - CV_PI);

    phase_unwrapping::HistogramPhaseUnwrapping::Params params;
    params.width = w;
    params.height = h;
    Ptr<phase_unwrapping::HistogramPhaseUnwrapping> pu =
        phase_unwrapping::HistogramPhaseUnwrapping::create(params);

    Mat unwrapped1, unwrapped2, rel1, rel2;
    pu->unwrapPhaseMap(phase, unwrapped1);
    pu->getInverseReliabilityMap(rel1);
    pu->unwrapPhaseMap(phase, unwrapped2);
    pu->getInverseReliabilityMap(rel2);

    EXPECT_EQ(0.0, cvtest::norm(unwrapped1, unwrapped2, NORM_INF));
    EXPECT_EQ(0.0, cvtest::norm(rel1, rel2, NORM_INF));
}

}} // namespace
