/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, Smart Engines Ltd, all rights reserved.
// Copyright (C) 2015, Institute for Information Transmission Problems of the Russian Academy of Sciences (Kharkevich Institute), all rights reserved.
// Copyright (C) 2015, Dmitry Nikolaev, Simon Karpenko, Michail Aliev, Elena Kuznetsova, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace cvtest
{
using namespace cv;
using namespace cv::ximgproc;
using namespace std;
using namespace testing;

using std::tr1::make_tuple;
using std::tr1::get;

//----------------------utils---------------------------------------------------

template <typename T> struct Eps
{
    static T get() { return 1; }
};
template <> struct Eps<float> { static float get() { return float(1e-3); } };
template <> struct Eps<double> { static double get() { return 1e-6; } };

template <typename T> struct MinPos
{
    static T get() { return Eps<T>::get(); }
};

template <typename T> struct Max { static T get()
{
    return saturate_cast<T>(numeric_limits<T>::max()); }
};

template <typename T> struct Rand
{
    static T get(T _min = MinPos<T>::get(), T _max = Max<T>::get())
    {
        RNG& rng = TS::ptr()->get_rng();
        return saturate_cast<T>(rng.uniform(int(std::max(MinPos<T>::get(),
                                                         _min)),
                                            int(std::min(Max<T>::get(),
                                                         _max))));
    }
};
template <> struct Rand <float>
{
    static float get(float _min = MinPos<float>::get(),
                     float _max = Max<float>::get())
    {
        RNG& rng = TS::ptr()->get_rng();
        return rng.uniform(std::max(MinPos<float>::get(), _min),
                           std::min(Max<float>::get(),    _max));
    }
};
template <> struct Rand <double>
{
    static double get(double _min = MinPos<double>::get(),
                     double _max = Max<double>::get())
    {
        RNG& rng = TS::ptr()->get_rng();
        return rng.uniform(std::max(MinPos<double>::get(), _min),
                           std::min(Max<double>::get(),    _max));
    }
};

template <typename T> struct Eq
{
    static bool get(T a, T b)
    {
        return a < b ? b - a < Eps<T>::get() : a - b < Eps<T>::get();
    }
};

//----------------------TestFHT-------------------------------------------------
class TestFHT
{
public:
    TestFHT() : ts(TS::ptr()) {}

    void run_n_tests(int depth,
                     int channels,
                     int pts_count,
                     int n_per_test);

private:
    template <typename T>
    int run_n_tests_t(int depth,
                      int channels,
                      int pts_count,
                      int n_per_test);

    template <typename T>
    int run_test(int depth,
                 int channels,
                 int pts_count);

    template <typename T>
    int put_random_points(Mat &img,
                          int count,
                          vector<Point> &pts);

    int run_func(Mat const&src,
                 Mat& fht);

    template <typename T>
    int validate_test_results(Mat const &fht,
                              Mat const &src,
                              vector<Point> const& pts);

    template <typename T> int validate_sum(Mat const& src, Mat const& fht);
    int validate_point(Mat const& fht, vector<Point> const &pts);
    int validate_line(Mat const& fht, Mat const& src, vector<Point> const& pts);

private:
    TS *ts;
};

template <typename T>
int TestFHT::put_random_points(Mat &img, int count, vector<Point> &pts)
{
    int code = TS::OK;

    pts.resize(count, Point(-1, -1));

    for (int i = 0; i < count; ++i)
    {
        RNG rng = ts->get_rng();
        Point const pt(rng.uniform(0, img.cols),
                       rng.uniform(0, img.rows));
        pts[i] = pt;

        for (int c = 0; c < img.channels(); ++c)
        {
            T color = Rand<T>::get(MinPos<T>::get(),
                                   T(Max<T>::get() / count));

            T *img_line = (T*)(img.data + img.step * pt.y);
            img_line[pt.x * img.channels() + c] = color;
        }
    }

    return code;
}

template <typename T>
int TestFHT::validate_sum(Mat const& src, Mat const& fht)
{
    int const channels = src.channels();
    if (fht.channels() != channels)
        return TS::FAIL_BAD_ARG_CHECK;


    vector<Mat> src_channels(channels);
    split(src, src_channels);
    vector<Mat> fht_channels(channels);
    split(fht, fht_channels);

    for (int c = 0; c < channels; ++c)
    {
        T const src_sum = saturate_cast<T>(sum(src_channels[c]).val[0]);
        for (int y = 0; y < fht.rows; ++y)
        {
            T const fht_sum = saturate_cast<T>(sum(fht_channels[c].row(y)).val[0]);
            if (!Eq<T>::get(src_sum, fht_sum))
            {
                ts->printf(TS::LOG,
                           "The sum of column #%d of channel #%d of the fast "
                           "hough transform result and the sum of source image"
                           " mismatch (=%g, should be =%g)\n",
                           y, c, (float)fht_sum, (float)src_sum);
                return TS::FAIL_BAD_ACCURACY;
            }
        }
    }
    return TS::OK;
}

int TestFHT::validate_point(Mat const& fht,
                            vector<Point> const &pts)
{
    if (pts.empty())
        return TS::OK;

    for (size_t i = 1; i < pts.size(); ++i)
    {
        if (pts[0] != pts[i])
            return TS::OK;
    }

    int const channels = fht.channels();
    vector<Mat> fht_channels(channels);
    split(fht, fht_channels);

    for (int c = 0; c < channels; ++c)
    {
        for (int y = 0; y < fht.rows; ++y)
        {
            int cnt = countNonZero(fht_channels[c].row(y));
            if (cnt != 1)
            {
                ts->printf(TS::LOG,
                           "The incorrect count of non-zero values in column "
                           "#%d, channel #%d of FastHoughTransform result "
                           "image (=%d, should be %d)\n",
                           y, c, cnt, 1);
                return TS::FAIL_BAD_ACCURACY;
            }
        }
    }
    return TS::OK;
}

static const double MAX_LDIST = 2.0;
int TestFHT::validate_line(Mat const& fht,
                           Mat const& src,
                           vector<Point> const& pts)
{
    size_t const size = (int)pts.size();
    if (size < 2)
        return TS::OK;
    size_t first_pt_i = 0, second_pt_i = 1;
    for (size_t i = first_pt_i + 1; i < size; ++i)
    {
        if (pts[i] != pts[first_pt_i])
        {
            second_pt_i = first_pt_i;
            break;
        }
    }
    if (pts[second_pt_i] == pts[first_pt_i])
        return TS::OK;
    for (size_t i = second_pt_i + 1; i < size; ++i)
    {
        if (pts[i] != pts[second_pt_i])
            return TS::OK;
    }
    const Point &f = pts[first_pt_i];
    const Point &s = pts[second_pt_i];

    int const channels = fht.channels();
    vector<Mat> fht_channels(channels);
    split(fht, fht_channels);

    for (int ch = 0; ch < channels; ++ch)
    {
        Point fht_max(-1, -1);
        minMaxLoc(fht_channels[ch], 0, 0, 0, &fht_max);
        Vec4i src_line =  HoughPoint2Line(fht_max, src,
                                          ARO_315_135, HDO_DESKEW, RO_STRICT);

        double const a = src_line[1] - src_line[3];
        double const b = src_line[2] - src_line[0];
        double const c = - (a * src_line[0] + b * src_line[1]);

        double const fd = abs(f.x * a + f.y * b + c) / sqrt(a * a + b * b);
        double const sd = abs(s.x * a + s.y * b + c) / sqrt(a * a + b * b);
        double const dist = std::max(fd, sd);

        if (dist > MAX_LDIST)
        {
            ts->printf(TS::LOG,
                       "Failed to detect max line in channels %d (distance "
                        "between point and line correspoinding of maximum in "
                        "FastHoughTransform space is #%g)\n", ch, dist);
            return TS::FAIL_BAD_ACCURACY;
        }
    }

    return TS::OK;
}

template <typename T>
int TestFHT::validate_test_results(Mat const &fht,
                                   Mat const &src,
                                   vector<Point> const& pts)
{
    int code = validate_sum<T>(src, fht);
    if (code == TS::OK)
        code = validate_point(fht, pts);
    if (code == TS::OK)
        code = validate_line(fht, src, pts);
    return code;
}

int TestFHT::run_func(Mat const&src,
                      Mat& fht)
{
    int code = TS::OK;
    FastHoughTransform(src, fht, src.depth());
    return code;
}

static Size random_size(int const max_size_log,
                        int const elem_size)
{
    RNG& rng = TS::ptr()->get_rng();
    return randomSize(rng, std::max(1,
        max_size_log - cvRound(log(double(elem_size)))));
}

static const int FHT_MAX_SIZE_LOG = 9;

template <typename T>
int TestFHT::run_test(int depth,
                      int channels,
                      int pts_count)
{
    int code = TS::OK;

    Size size = random_size(FHT_MAX_SIZE_LOG,
                            CV_ELEM_SIZE(CV_MAKE_TYPE(depth, channels)));
    Mat src = Mat::zeros(size, CV_MAKETYPE(depth, channels));

    vector<Point> pts;
    code = put_random_points<T>(src, pts_count, pts);
    if (code != TS::OK)
        return code;

    Mat fht;
    code = run_func(src, fht);
    if (code != TS::OK)
        return code;

    code = validate_test_results<T>(fht, src, pts);
    return code;
}

void TestFHT::run_n_tests(int depth,
                          int channels,
                          int pts_count,
                          int  n)
{
    try
    {
        int code = TS::OK;

        switch (depth)
        {
        case CV_8U:
            code = run_n_tests_t<uchar>(depth, channels, pts_count, n);
            break;
        case CV_8S:
            code = run_n_tests_t<schar>(depth, channels, pts_count, n);
            break;
        case CV_16U:
            code = run_n_tests_t<ushort>(depth, channels, pts_count, n);
            break;
        case CV_16S:
            code = run_n_tests_t<short>(depth, channels, pts_count, n);
            break;
        case CV_32S:
            code = run_n_tests_t<int>(depth, channels, pts_count, n);
            break;
        case CV_32F:
            code = run_n_tests_t<float>(depth, channels, pts_count, n);
            break;
        case CV_64F:
            code = run_n_tests_t<double>(depth, channels, pts_count, n);
            break;
        default:
            code = TS::FAIL_BAD_ARG_CHECK;
            ts->printf(TS::LOG, "Unknown depth %d\n", depth);
            break;
        }
        if (code != TS::OK)
            throw TS::FailureCode(code);
    }
    catch (const TS::FailureCode& fc)
    {
        std::string errorStr = TS::str_from_code(fc);
        ts->printf(TS::LOG,
                   "General failure:\n\t%s (%d)\n", errorStr.c_str(), fc);

        ts->set_failed_test_info(fc);
    }
    catch(...)
    {
        ts->printf(TS::LOG, "Unknown failure\n");
        ts->set_failed_test_info(TS::FAIL_EXCEPTION);
    }
}

template <typename T>
int TestFHT::run_n_tests_t(int depth,
                           int channels,
                           int pts_count,
                           int n)
{
    int code = TS::OK;
    for (int iTest = 0; iTest < n; ++iTest)
    {
        code = run_test<T>(depth, channels, pts_count);
        if (code != TS::OK)
        {
            ts->printf(TS::LOG, "Test %d failed with code %d\n", iTest, code);
            break;
        }
    }
    return code;
}

//----------------------TEST_P--------------------------------------------------
typedef std::tr1::tuple<int, int, int, int> Depth_Channels_PtsC_nPerTest;
typedef TestWithParam<Depth_Channels_PtsC_nPerTest> FastHoughTransformTest;

TEST_P(FastHoughTransformTest, accuracy)
{
    int  const depth      = get<0>(GetParam());
    int  const channels   = get<1>(GetParam());
    int  const pts_count  = get<2>(GetParam());
    int  const n_per_test = get<3>(GetParam());

    TestFHT testFht;
    testFht.run_n_tests(depth, channels, pts_count, n_per_test);
}

#define FHT_ALL_DEPTHS CV_8U, CV_16U, CV_32S, CV_32F, CV_64F
#define FHT_ALL_CHANNELS 1, 3, 4

INSTANTIATE_TEST_CASE_P(FullSet, FastHoughTransformTest,
                        Combine(Values(FHT_ALL_DEPTHS),
                                Values(FHT_ALL_CHANNELS),
                                Values(1, 2),
                                Values(5)));

#undef FHT_ALL_DEPTHS
#undef FHT_ALL_CHANNELS

} // namespace cvtest
