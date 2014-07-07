#include "test_precomp.hpp"

#define NO_COMPARISON

namespace cvtest
{
    TEST(xphoto_dctimagedenoising, regression)
    {
        cv::String dir = cvtest::TS::ptr()->get_data_path() + "dct_image_denoising/";
        int nTests = 1;

        float psnrThreshold[] = {0.5};

        int psize[] = {8};
        double sigma[] = {9.0};

        for (int i = 0; i < nTests; ++i)
        {
            cv::String srcName = dir + cv::format( "sources/%02d.png", i + 1);
            cv::Mat src = cv::imread( srcName, 1 );

            cv::String previousResultName = dir + cv::format( "results/%02d.png", i + 1 );
            cv::Mat previousResult = cv::imread( previousResultName, 1 );

            cv::Mat sqrError = ( src - previousResult ).mul( src - previousResult );
            cv::Scalar mse = cv::sum(sqrError) / cv::Scalar::all( sqrError.total()*sqrError.channels() );
            double psnr = 10*log10(3*255*255/(mse[0] + mse[1] + mse[2]));


            cv::Mat currentResult, fastNlMeansResult;

#ifndef NO_COMPARISON
            double currentTime = clock() / double(CLOCKS_PER_SEC);
#endif
            cv::dctDenoising(src, currentResult, sigma[i], psize[i]);
#ifndef NO_COMPARISON
            currentTime = clock() / double(CLOCKS_PER_SEC) - currentTime;
            std::cout << "---- dct denoising time = " << currentTime << " (sec) ----" << std::endl;
#endif

            cv::Mat sqrError1 = ( currentResult - previousResult )
                .mul( currentResult - previousResult );
            cv::Scalar mse1 = cv::sum(sqrError1) / cv::Scalar::all( sqrError1.total()*sqrError1.channels() );
            double psnr1 = 10*log10(3*255*255/(mse1[0] + mse1[1] + mse1[2])) - psnr;
#ifndef NO_COMPARISON
            std::cout << "---- dct PSNR rate = " << psnr1 << " ----" << std::endl;
#endif

#ifndef NO_COMPARISON
            double fastNlMeansTime = clock() / double(CLOCKS_PER_SEC);

            if ( src.channels() == 3 )
                cv::fastNlMeansDenoisingColored(src, fastNlMeansResult);
            else if ( src.channels() == 1 )
                cv::fastNlMeansDenoising(src, fastNlMeansResult);

            fastNlMeansTime = clock() / double(CLOCKS_PER_SEC) - fastNlMeansTime;
#ifdef NO_COMPARISON
            std::cout << "---- nonlocal means denoising time = " << fastNlMeansTime << " (sec) ----" << std::endl;
#endif

            cv::Mat sqrError2 = ( fastNlMeansResult - previousResult )
                .mul( fastNlMeansResult - previousResult );
            cv::Scalar mse2 = cv::sum(sqrError2) / cv::Scalar::all( sqrError2.total()*sqrError2.channels() );
            double psnr2 = 10*log10(3*255*255/(mse2[0] + mse2[1] + mse2[2])) - psnr;
            std::cout << "---- nonlocal means PSNR rate = " << psnr2 << " ----" << std::endl;
#endif

            EXPECT_GE( psnr1, psnrThreshold[i] );
        }
    }
}