/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_ALLOCATOR_HPP
#define OPENCV_FASTCV_ALLOCATOR_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
    * @brief Gets the default FastCV allocator.
    * This function returns a pointer to the default FastCV allocator, which is optimized
    * for use with DSP.
    *
    * @return Pointer to the default FastCV allocator.
    */
CV_EXPORTS cv::MatAllocator* getDefaultFastCVAllocator();
//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_ALLOCATOR_HPP
