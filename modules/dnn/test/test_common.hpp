#ifndef __OPENCV_TEST_COMMON_HPP__
#define __OPENCV_TEST_COMMON_HPP__

inline const std::string &getOpenCVExtraDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

inline void normAssert(cv::InputArray ref, cv::InputArray get, const char *comment = "")
{
    double normL1 = cvtest::norm(ref, get, cv::NORM_L1)/ ref.getMat().total();
    EXPECT_NEAR(normL1, 0, 0.0001) << comment;

    double normInf = cvtest::norm(ref, get, cv::NORM_INF);
    EXPECT_NEAR(normInf, 0, 0.001) << comment;
}

inline void normAssert(cv::dnn::Blob &ref, cv::dnn::Blob &test, const char *comment = "")
{
    ASSERT_EQ(ref.shape(), test.shape());
    normAssert(ref.getMatRef(), test.getMatRef(), comment);
}

#endif
