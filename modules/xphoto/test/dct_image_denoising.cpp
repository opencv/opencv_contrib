#include "test_precomp.hpp"

namespace cvtest
{
    TEST(xphoto_dctimagedenoising, regression)
    {
        cv::String dir = cvtest::TS::ptr()->get_data_path() + "dct_image_denoising/";
        int nTests = 12;

		int psize = 3.0;
		float psnrThreshold = 40.0;
		float sigma = 15.0;

        for (int i = 0; i < nTests; ++i)
        {
            cv::String srcName = dir + cv::format( "sources/%02d.png", i + 1);
            cv::Mat src = cv::imread( srcName );

            cv::String previousResultName = dir + cv::format( "results/%02d.png", i + 1 );
            cv::Mat previousResult = cv::imread( previousResultName, 1 );

            cv::Mat currentResult;
            cv::dctDenoising(src, currentResult, sigma, psize);

            cv::Mat sqrError = ( currentResult - previousResult )
                .mul( currentResult - previousResult );
            cv::Scalar mse = cv::sum(sqrError) / cv::Scalar::all( sqrError.total()*sqrError.channels() );
			double psnr = 10*log10(3*255*255/(mse[0] + mse[1] + mse[2]));

            EXPECT_GE( psnr, psnrThreshold );
        }
    }
}