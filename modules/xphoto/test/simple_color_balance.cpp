#include "test_precomp.hpp"

namespace cvtest
{
    TEST(xphoto_simplecolorbalance, regression)
    {
        cv::String dir = cvtest::TS::ptr()->get_data_path() + "cv/xphoto/simple_white_balance/";
        int nTests = 12;
        float threshold = 0.005f;
        cv::Ptr<cv::xphoto::WhiteBalancer> wb = cv::xphoto::createSimpleWB();

        for (int i = 0; i < nTests; ++i)
        {
            cv::String srcName = dir + cv::format( "sources/%02d.png", i + 1);
            cv::Mat src = cv::imread( srcName, 1 );
            ASSERT_TRUE(!src.empty());

            cv::String previousResultName = dir + cv::format( "results/%02d.png", i + 1 );
            cv::Mat previousResult = cv::imread( previousResultName, 1 );

            cv::Mat currentResult;
            wb->balanceWhite(src, currentResult);

            cv::Mat sqrError = ( currentResult - previousResult )
                .mul( currentResult - previousResult );
            cv::Scalar mse = cv::sum(sqrError) / cv::Scalar::all( double( sqrError.total()*sqrError.channels() ) );

            EXPECT_LE( mse[0]+mse[1]+mse[2]+mse[3], threshold );
        }
    }
}
