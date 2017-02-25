#include "test_precomp.hpp"

namespace cvtest {

    using namespace cv;

    void ref_autowbGrayworld(InputArray _src, OutputArray _dst, float thresh)
    {
        Mat src = _src.getMat();

        _dst.create(src.size(), src.type());
        Mat dst = _dst.getMat();

        int width  = src.cols,
            height = src.rows,
            N      = width*height,
            N3     = N*3;

        // Calculate sum of pixel values of each channel
        const uchar* src_data = src.ptr<uchar>(0);
        unsigned long sum1 = 0, sum2 = 0, sum3 = 0;
        int i = 0;
        unsigned int minRGB, maxRGB, thresh255 = cvRound(thresh * 255);
        for ( ; i < N3; i += 3 )
        {
            minRGB = std::min(src_data[i], std::min(src_data[i + 1], src_data[i + 2]));
            maxRGB = std::max(src_data[i], std::max(src_data[i + 1], src_data[i + 2]));
            if ( (maxRGB - minRGB) * 255 > thresh255 * maxRGB ) continue;
            sum1 += src_data[i];
            sum2 += src_data[i + 1];
            sum3 += src_data[i + 2];
        }

        // Find inverse of averages
        double inv1 = sum1 == 0 ? 0.f : (double)N / (double)sum1,
               inv2 = sum2 == 0 ? 0.f : (double)N / (double)sum2,
               inv3 = sum3 == 0 ? 0.f : (double)N / (double)sum3;

        // Find maximum
        double inv_max = std::max(std::max(inv1, inv2), inv3);

        // Scale by maximum
        if ( inv_max > 0 )
        {
            inv1 = (double) inv1 / inv_max;
            inv2 = (double) inv2 / inv_max;
            inv3 = (double) inv3 / inv_max;
        }

        // Fixed point arithmetic, mul by 2^8 then shift back 8 bits
        int i_inv1 = cvRound(inv1 * (1 << 8)),
            i_inv2 = cvRound(inv2 * (1 << 8)),
            i_inv3 = cvRound(inv3 * (1 << 8));

        // Scale input pixel values
        uchar* dst_data = dst.ptr<uchar>(0);
        i = 0;
        for ( ; i < N3; i += 3 )
        {
            dst_data[i]     = (uchar)((src_data[i]     * i_inv1) >> 8);
            dst_data[i + 1] = (uchar)((src_data[i + 1] * i_inv2) >> 8);
            dst_data[i + 2] = (uchar)((src_data[i + 2] * i_inv3) >> 8);
        }
    }

    TEST(xphoto_grayworld_white_balance, regression)
    {
        String dir = cvtest::TS::ptr()->get_data_path() + "cv/xphoto/simple_white_balance/";
        const int nTests = 8;
        const float wb_thresh = 0.5f;
        const float acc_thresh = 2.f;
        Ptr<xphoto::GrayworldWB> wb = xphoto::createGrayworldWB();
        wb->setSaturationThreshold(wb_thresh);

        for ( int i = 0; i < nTests; ++i )
        {
            String srcName = dir + format("sources/%02d.png", i + 1);
            Mat src = imread(srcName, IMREAD_COLOR);
            ASSERT_TRUE(!src.empty());

            Mat referenceResult;
            ref_autowbGrayworld(src, referenceResult, wb_thresh);

            Mat currentResult;
            wb->balanceWhite(src, currentResult);
            ASSERT_LE(cv::norm(currentResult, referenceResult, NORM_INF), acc_thresh);

            // test the 16-bit depth:
            Mat currentResult_16U, src_16U;
            src.convertTo(src_16U, CV_16UC3, 256.0);
            wb->balanceWhite(src_16U, currentResult_16U);
            currentResult_16U.convertTo(currentResult, CV_8UC3, 1/256.0);
            ASSERT_LE(cv::norm(currentResult, referenceResult, NORM_INF), acc_thresh);
        }
    }

}
