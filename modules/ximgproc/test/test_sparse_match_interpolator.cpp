/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#include "test_precomp.hpp"
#include "opencv2/ximgproc/sparse_match_interpolator.hpp"
#include <fstream>

namespace cvtest
{

using namespace std;
using namespace std::tr1;
using namespace testing;
using namespace perf;
using namespace cv;
using namespace cv::ximgproc;

static string getDataDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

const float FLOW_TAG_FLOAT = 202021.25f;
Mat readOpticalFlow( const String& path )
{
//    CV_Assert(sizeof(float) == 4);
    //FIXME: ensure right sizes of int and float - here and in writeOpticalFlow()

    Mat_<Point2f> flow;
    ifstream file(path.c_str(), std::ios_base::binary);
    if ( !file.good() )
        return flow; // no file - return empty matrix

    float tag;
    file.read((char*) &tag, sizeof(float));
    if ( tag != FLOW_TAG_FLOAT )
        return flow;

    int width, height;

    file.read((char*) &width, 4);
    file.read((char*) &height, 4);

    flow.create(height, width);

    for ( int i = 0; i < flow.rows; ++i )
    {
        for ( int j = 0; j < flow.cols; ++j )
        {
            Point2f u;
            file.read((char*) &u.x, sizeof(float));
            file.read((char*) &u.y, sizeof(float));
            if ( !file.good() )
            {
                flow.release();
                return flow;
            }

            flow(i, j) = u;
        }
    }
    file.close();
    return flow;
}

CV_ENUM(GuideTypes, CV_8UC1, CV_8UC3)
typedef tuple<Size, GuideTypes> InterpolatorParams;
typedef TestWithParam<InterpolatorParams> InterpolatorTest;

TEST(InterpolatorTest, ReferenceAccuracy)
{
    double MAX_DIF = 1.0;
    double MAX_MEAN_DIF = 1.0 / 256.0;
    string dir = getDataDir() + "cv/sparse_match_interpolator";

    Mat src = imread(getDataDir() + "cv/optflow/RubberWhale1.png",IMREAD_COLOR);
    ASSERT_FALSE(src.empty());

    Mat ref_flow = readOpticalFlow(dir + "/RubberWhale_reference_result.flo");
    ASSERT_FALSE(ref_flow.empty());

    ifstream file((dir + "/RubberWhale_sparse_matches.txt").c_str());
    float from_x,from_y,to_x,to_y;
    vector<Point2f> from_points;
    vector<Point2f> to_points;

    while(file >> from_x >> from_y >> to_x >> to_y)
    {
        from_points.push_back(Point2f(from_x,from_y));
        to_points.push_back(Point2f(to_x,to_y));
    }

    cv::setNumThreads(cv::getNumberOfCPUs());
    Mat res_flow;

    Ptr<EdgeAwareInterpolator> interpolator = createEdgeAwareInterpolator();
    interpolator->setK(128);
    interpolator->setSigma(0.05f);
    interpolator->setUsePostProcessing(true);
    interpolator->setFGSLambda(500.0f);
    interpolator->setFGSSigma(1.5f);
    interpolator->interpolate(src,from_points,Mat(),to_points,res_flow);

    EXPECT_LE(cv::norm(res_flow, ref_flow, NORM_INF), MAX_DIF);
    EXPECT_LE(cv::norm(res_flow, ref_flow, NORM_L1) , MAX_MEAN_DIF*res_flow.total());
}

TEST_P(InterpolatorTest, MultiThreadReproducibility)
{
    if (cv::getNumberOfCPUs() == 1)
        return;

    double MAX_DIF = 1.0;
    double MAX_MEAN_DIF = 1.0 / 256.0;
    int loopsCount = 2;
    RNG rng(0);

    InterpolatorParams params = GetParam();
    Size size       = get<0>(params);
    int guideType   = get<1>(params);

    Mat from(size, guideType);
    randu(from, 0, 255);

    int num_matches = rng.uniform(5,SHRT_MAX-1);
    vector<Point2f> from_points;
    vector<Point2f> to_points;

    for(int i=0;i<num_matches;i++)
    {
        from_points.push_back(Point2f(rng.uniform(0.01f,(float)size.width-1.01f),rng.uniform(0.01f,(float)size.height-1.01f)));
        to_points.push_back(Point2f(rng.uniform(0.01f,(float)size.width-1.01f),rng.uniform(0.01f,(float)size.height-1.01f)));
    }

    for (int iter = 0; iter <= loopsCount; iter++)
    {
        int K = rng.uniform(4,512);
        float sigma = rng.uniform(0.01f,0.5f);
        float FGSlambda = rng.uniform(100.0f, 10000.0f);
        float FGSsigma  = rng.uniform(0.5f, 100.0f);

        Ptr<EdgeAwareInterpolator> interpolator = createEdgeAwareInterpolator();
        interpolator->setK(K);
        interpolator->setSigma(sigma);
        interpolator->setUsePostProcessing(true);
        interpolator->setFGSLambda(FGSlambda);
        interpolator->setFGSSigma(FGSsigma);

        cv::setNumThreads(cv::getNumberOfCPUs());
        Mat resMultiThread;
        interpolator->interpolate(from,from_points,Mat(),to_points,resMultiThread);

        cv::setNumThreads(1);
        Mat resSingleThread;
        interpolator->interpolate(from,from_points,Mat(),to_points,resSingleThread);

        EXPECT_LE(cv::norm(resSingleThread, resMultiThread, NORM_INF), MAX_DIF);
        EXPECT_LE(cv::norm(resSingleThread, resMultiThread, NORM_L1) , MAX_MEAN_DIF*resMultiThread.total());
    }
}
INSTANTIATE_TEST_CASE_P(FullSet,InterpolatorTest, Combine(Values(szODD,szVGA), GuideTypes::all()));
}