// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef OPENCV_SIGNAL_SIGNAL_RESAMPLE_HPP
#define OPENCV_SIGNAL_SIGNAL_RESAMPLE_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace signal {

//! @addtogroup signal
//! @{

/** @brief Signal resampling
 *
 * @param[in]  inputSignal              Array with input signal.
 * @param[out] outSignal                Array with output signal
 * @param[in]  inFreq                   Input signal frequency.
 * @param[in]  outFreq                  Output signal frequency.
 * Signal resampling implemented a cubic interpolation function and a filtering function based on Kaiser window and Bessel function, used to construct a FIR filter.
 * Result is similar to `scipy.signal.resample`.

Detail: https://en.wikipedia.org/wiki/Sample-rate_conversion
*/
CV_EXPORTS_W void resampleSignal(InputArray inputSignal, OutputArray outSignal, const int inFreq, const int outFreq);

//! @}

}
}
#endif
