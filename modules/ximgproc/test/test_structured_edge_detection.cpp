// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ximgproc_StructuredEdgeDetection, regression)
{
    cv::String subfolder = "cv/ximgproc/";
    cv::String dir = cvtest::TS::ptr()->get_data_path() + subfolder;
    int nTests = 12;
    float threshold = 0.01f;
    
    cv::String modelName = dir + "model.yml.gz";
    cv::Ptr<cv::ximgproc::StructuredEdgeDetection> pDollar =
        cv::ximgproc::createStructuredEdgeDetection(modelName);

    for (int i = 0; i < nTests; ++i)
    {
        cv::String srcName = dir + cv::format( "sources/%02d.png", i + 1);
        cv::Mat src = cv::imread( srcName, 1 );
        ASSERT_TRUE(!src.empty());

        cv::String previousResultName = dir + cv::format( "results/%02d.png", i + 1 );
        cv::Mat previousResult = cv::imread( previousResultName, 0 );
        previousResult.convertTo( previousResult, CV_32F, 1/255.0 );

        src.convertTo( src, CV_32F, 1/255.0 );

        cv::Mat currentResult( src.size(), src.type() );
        pDollar->detectEdges( src, currentResult );

        cv::Mat sqrError = ( currentResult - previousResult )
            .mul( currentResult - previousResult );
        cv::Scalar mse = cv::sum(sqrError) / cv::Scalar::all( double( sqrError.total() ) );

        EXPECT_LE( mse[0], threshold );
    }
}

TEST(ximgproc_EdgeBoxes, DetectorReuse)
{
    cv::Mat edge1 = cv::Mat::zeros(128, 128, CV_32FC1);
    cv::Mat orient1 = cv::Mat::zeros(128, 128, CV_32FC1);
    cv::rectangle(edge1, cv::Rect(20, 20, 60, 60), cv::Scalar(0.9f), 1);
    cv::rectangle(edge1, cv::Rect(35, 35, 30, 30), cv::Scalar(0.8f), 1);

    cv::Mat edge2 = cv::Mat::zeros(128, 128, CV_32FC1);
    cv::Mat orient2 = cv::Mat::zeros(128, 128, CV_32FC1);
    cv::rectangle(edge2, cv::Rect(15, 25, 70, 50), cv::Scalar(0.85f), 1);
    cv::rectangle(edge2, cv::Rect(40, 10, 45, 80), cv::Scalar(0.25f), 1);

    cv::Ptr<cv::ximgproc::EdgeBoxes> ebReused = cv::ximgproc::createEdgeBoxes();
    std::vector<cv::Rect> boxes1, boxes2Reused, boxes2Fresh;
    cv::Mat scores1, scores2Reused, scores2Fresh;
    ebReused->getBoundingBoxes(edge1, orient1, boxes1, scores1);
    ebReused->getBoundingBoxes(edge2, orient2, boxes2Reused, scores2Reused);

    cv::Ptr<cv::ximgproc::EdgeBoxes> ebFresh = cv::ximgproc::createEdgeBoxes();
    ebFresh->getBoundingBoxes(edge2, orient2, boxes2Fresh, scores2Fresh);

    ASSERT_EQ(boxes2Fresh.size(), boxes2Reused.size());
    for (size_t i = 0; i < boxes2Fresh.size(); ++i)
        EXPECT_EQ(boxes2Fresh[i], boxes2Reused[i]);
    EXPECT_EQ(0.0, cv::norm(scores2Fresh, scores2Reused, cv::NORM_INF));
}

}} // namespace
