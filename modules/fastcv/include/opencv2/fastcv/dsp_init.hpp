/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_DSP_INIT_HPP
#define OPENCV_FASTCV_DSP_INIT_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {
namespace dsp {

    //! @addtogroup fastcv
    //! @{

    /**
     * @brief Initializes the FastCV DSP environment.
     * 
     * This function sets up the necessary environment for FastCV DSP operations.
     * It ensures that the DSP context is initialized and sets the custom memory allocator
     * for FastCV operations.
     * 
     * @return int Returns 0 on success, and a non-zero value on failure.
     */
    CV_EXPORTS_W int fastcvq6init();

    /**
     * @brief Deinitializes the FastCV DSP environment.
     * 
     * This function cleans up the FastCV DSP environment, releasing any resources
     * that were allocated during initialization.
     */
    CV_EXPORTS_W void fastcvq6deinit();
    //! @}

} // dsp::
} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_DSP_INIT_HPP
