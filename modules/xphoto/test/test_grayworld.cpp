#include "test_precomp.hpp"

namespace cvtest {

    using namespace cv;

    void ref_autowbGrayworld(InputArray _src, OutputArray _dst, const float thresh)
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
        double minRGB, maxRGB, satur;
        for ( ; i < N3; i += 3 )
        {
            minRGB = std::min(src_data[i], std::min(src_data[i + 1], src_data[i + 2]));
            maxRGB = std::max(src_data[i], std::max(src_data[i + 1], src_data[i + 2]));
            satur = (maxRGB - minRGB) / maxRGB;
            if ( satur > thresh ) continue;
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

        // Scale input pixel values
        uchar* dst_data = dst.ptr<uchar>(0);
        i = 0;
        for ( ; i < N3; i += 3 )
        {
            dst_data[i]     = src_data[i]     * inv1;
            dst_data[i + 1] = src_data[i + 1] * inv2;
            dst_data[i + 2] = src_data[i + 2] * inv3;
        }
    }

    TEST(xphoto_grayworld_white_balance, regression)
    {
        String subfolder = "/xphoto/";
        String dir = cvtest::TS::ptr()->get_data_path() + subfolder + "simple_white_balance/";
        const int nTests = 14;
        const float wb_thresh = 0.5f;
        const float acc_thresh = 2.f;

        for ( int i = 0; i < nTests; ++i )
        {
            String srcName = dir + format("sources/%02d.png", i + 1);
            Mat src = imread(srcName, IMREAD_COLOR);
            ASSERT_TRUE(!src.empty());

            Mat referenceResult;
            ref_autowbGrayworld(src, referenceResult, wb_thresh);

            Mat currentResult;
            xphoto::autowbGrayworld(src, currentResult, wb_thresh);

            ASSERT_LE(cv::norm(currentResult, referenceResult, NORM_INF), acc_thresh);
        }
    }

}
