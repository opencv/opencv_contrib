#ifndef CVVISUAL_DEBUG_SHOW_IMAGE_HPP
#define CVVISUAL_DEBUG_SHOW_IMAGE_HPP

#include <string>

#include "opencv2/core.hpp"

#include "call_meta_data.hpp"
#include "debug_mode.hpp"

#ifdef CV_DOXYGEN
#define CVVISUAL_DEBUGMODE
#endif

namespace cvv
{

//! @addtogroup cvv
//! @{

namespace impl
{
// implementation outside API
CV_EXPORTS void showImage(cv::InputArray img, const CallMetaData &data,
               const char *description, const char *view);
} // namespace impl

#ifdef CVVISUAL_DEBUGMODE
/** @brief Add a single image to debug GUI (similar to imshow \<\>).

@param img Image to show in debug GUI.
@param metaData Properly initialized CallMetaData struct, i.e. information about file, line and
function name for GUI. Use CVVISUAL_LOCATION macro.
@param description Human readable description to provide context to image.
@param view Preselect view that will be used to visualize this image in GUI. Other views can still
be selected in GUI later on.
 */
static inline void showImage(cv::InputArray img,
                             impl::CallMetaData metaData = impl::CallMetaData(),
                             const char *description = nullptr,
                             const char *view = nullptr)
{
	if (debugMode())
	{
		impl::showImage(img, metaData, description, view);
	}
}
/** @overload */
static inline void showImage(cv::InputArray img, impl::CallMetaData metaData,
                             const ::std::string &description,
                             const ::std::string &view = "")
{
	if (debugMode())
	{
		impl::showImage(img, metaData, description.c_str(),
		                view.c_str());
	}
}
#else
static inline void showImage(cv::InputArray,
                             impl::CallMetaData = impl::CallMetaData(),
                             const char * = nullptr, const char * = nullptr)
{
}
static inline void showImage(cv::InputArray, impl::CallMetaData,
                             const ::std::string &, const ::std::string &)
{
}
#endif

//! @}

} // namespace cvv

#endif
