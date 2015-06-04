#include "test_precomp.hpp"

namespace cvtest {

    using namespace cv;

    // TODO: Remove debug print statements
    void ref_autowbGrayworld(InputArray _src, OutputArray _dst)
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
        for (; i < N3; i += 3)
        {
            sum1 += src_data[i + 0];
            sum2 += src_data[i + 1];
            sum3 += src_data[i + 2];
        }
        //printf("sums:\t\t\t%lu, %lu, %lu\n", sum1, sum2, sum3);

        // Find inverse of averages
        double inv1 = sum1 == 0 ? 0.f : (double)N / (double)sum1,
               inv2 = sum2 == 0 ? 0.f : (double)N / (double)sum2,
               inv3 = sum3 == 0 ? 0.f : (double)N / (double)sum3;
        //printf("inverse avgs:\t\t%f, %f, %f\n", inv1, inv2, inv3);

        // Find maximum
        double inv_max = std::max(std::max(inv1, inv2), inv3);

        // Scale by maximum
        if (inv_max > 0)
        {
            inv1 = (double) inv1 / inv_max;
            inv2 = (double) inv2 / inv_max;
            inv3 = (double) inv3 / inv_max;
        }
        //printf("scaling factors:\t%f, %f, %f\n", inv1, inv2, inv3);
        //printf("scaling factors applied:\t%f, %f, %f\n",
        //        (double) sum1 * inv1,
        //        (double) sum2 * inv2,
        //        (double) sum3 * inv3);

        // Scale input pixel values
        uchar* dst_data = dst.ptr<uchar>(0);
        i = 0;
        for (; i < N3; i += 3)
        {
            dst_data[i + 0] = src_data[i + 0] * inv1;
            dst_data[i + 1] = src_data[i + 1] * inv2;
            dst_data[i + 2] = src_data[i + 2] * inv3;
        }
        //imshow("original", src);
        //imshow("grayworld", dst);
        //waitKey();
    }

    TEST(xphoto_grayworld_white_balance, regression)
    {
        String subfolder = "/xphoto/";
        String dir = cvtest::TS::ptr()->get_data_path() + subfolder + "simple_white_balance/";
        int nTests = 14;
        float threshold = 2.f;

        for (int i = 0; i < nTests; ++i)
        {
            String srcName = dir + format("sources/%02d.png", i + 1);
            Mat src = imread(srcName, IMREAD_COLOR);
            ASSERT_TRUE(!src.empty());

            Mat referenceResult;
            ref_autowbGrayworld(src, referenceResult);

            Mat currentResult;
            xphoto::autowbGrayworld(src, currentResult);

            ASSERT_LE(cv::norm(currentResult, referenceResult, NORM_INF), threshold);
        }
    }

}
