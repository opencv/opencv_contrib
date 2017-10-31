// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef __OPENCV_TRACKING_UTILS_HPP__

#include "precomp.hpp"
#include <algorithm>

namespace cv {
namespace tracking_internal
{
/** Computes normalized corellation coefficient between the two patches (they should be
* of the same size).*/
    double computeNCC(const Mat& patch1, const Mat& patch2);

    template<typename T>
    T getMedianAndDoPartition(std::vector<T>& values)
    {
        size_t size = values.size();
        if(size%2==0)
        {
            std::nth_element(values.begin(), values.begin() + size/2-1, values.end());
            T firstMedian = values[size/2-1];

            std::nth_element(values.begin(), values.begin() + size/2, values.end());
            T secondMedian = values[size/2];

            return (firstMedian + secondMedian) / (T)2;
        }
        else
        {
            size_t medianIndex = (size - 1) / 2;
            std::nth_element(values.begin(), values.begin() + medianIndex, values.end());

            return values[medianIndex];
        }
    }

    template<typename T>
    T getMedian(const std::vector<T>& values)
    {
        std::vector<T> copy(values);
        return getMedianAndDoPartition(copy);
    }
}
}
#endif
