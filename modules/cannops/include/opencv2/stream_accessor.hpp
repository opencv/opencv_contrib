// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANNOPS_STREAM_ACCESSOR_HPP
#define OPENCV_CANNOPS_STREAM_ACCESSOR_HPP

#include <acl/acl_base.h>
#include "opencv2/cann.hpp"

namespace cv
{
namespace cann
{
//! @addtogroup cann_struct
//! @{

/** @brief Class that enables getting aclrtAscendStream from cann::AscendStream
 */
struct AscendStreamAccessor
{
    CV_EXPORTS static aclrtStream getStream(const AscendStream& stream);
    CV_EXPORTS static AscendStream wrapStream(aclrtStream stream);
};

/** @brief Class that enables getting aclrtAscendEvent from cann::AscendEvent
 */
struct AscendEventAccessor
{
    CV_EXPORTS static aclrtEvent getEvent(const AscendEvent& event);
    CV_EXPORTS static AscendEvent wrapEvent(aclrtEvent event);
};

//! @} cann_struct

} // namespace cann
} // namespace cv

#endif // OPENCV_CANNOPS_STREAM_ACCESSOR_HPP
