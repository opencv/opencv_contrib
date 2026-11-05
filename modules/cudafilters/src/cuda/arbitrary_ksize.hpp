// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Shared by row_filter.hpp and column_filter.hpp for the kernels that take ksize at run time.

#ifndef OPENCV_CUDAFILTERS_ARBITRARY_KSIZE_HPP
#define OPENCV_CUDAFILTERS_ARBITRARY_KSIZE_HPP

#include "opencv2/core/cuda/common.hpp"

namespace filter
{
    // One launch can hold this many blocks along y. The run-time kernels step through the rows beyond it.
    const int MAX_GRID_Y = 65535;

    // True when the kernel reaches a full image length or more past an edge. The Brd classes used by
    // the fixed-size kernels mirror only one period, so BORDER_REFLECT and BORDER_REFLECT101 go wrong there.
    inline bool kernelOutreachesImage(int ksize, int anchor, int len)
    {
        return anchor >= len || ksize - anchor > len;
    }

    // Index that position p maps to on a line of len elements, for p any distance outside it.
    template <int BRD_TYPE>
    __device__ __forceinline__ int borderIdx(int p, int len)
    {
        if (BRD_TYPE == cv::BORDER_REPLICATE)
            return p < 0 ? 0 : (p >= len ? len - 1 : p);

        if (BRD_TYPE == cv::BORDER_WRAP)
        {
            p %= len;
            if (p < 0)
                p += len;
            return p;
        }

        if (len == 1)
            return 0;

        if (BRD_TYPE == cv::BORDER_REFLECT)
        {
            const int period = 2 * len;
            p %= period;
            if (p < 0)
                p += period;
            if (p >= len)
                p = period - 1 - p;
            return p;
        }

        // cv::BORDER_REFLECT101
        const int period = 2 * (len - 1);
        p %= period;
        if (p < 0)
            p += period;
        if (p >= len)
            p = period - p;
        return p;
    }
}

#endif
