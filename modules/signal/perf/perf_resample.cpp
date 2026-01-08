// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

namespace opencv_test { namespace {

typedef TestBaseWithParam< tuple<uint32_t, uint32_t, uint32_t> > TestResampleFunc;

PERF_TEST_P( TestResampleFunc, resample_sin_signal,
             testing::Combine(
                testing::Values(1234U, 12345U, 123456U, 1234567U, 12345678U),
                testing::Values(16000U, 32000U, 44100U, 48000U),
                testing::Values(48000U, 44100U, 32000U, 16000U))
)
{
    uint32_t sample_signal_size = GET_PARAM(0);
    uint32_t inFreq = GET_PARAM(1);
    uint32_t outFreq = GET_PARAM(2);

    Mat1f sample_signal(Size(sample_signal_size,1U));
    Mat1f outSignal(Size(1U, 1U));
    for (uint32_t i = 0U; i < (uint32_t)sample_signal.cols; ++i)
    {
        sample_signal.at<float>(0, i) = sinf(float(i));
    }
    declare.in(sample_signal).out(outSignal);
    TEST_CYCLE() resampleSignal(sample_signal, outSignal, inFreq, outFreq);
    SANITY_CHECK_NOTHING();
}

}}
