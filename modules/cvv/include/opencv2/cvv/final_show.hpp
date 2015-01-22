#ifndef CVVISUAL_FINAL_SHOW_HPP
#define CVVISUAL_FINAL_SHOW_HPP

#include "opencv2/core.hpp"
#include "debug_mode.hpp"

namespace cvv
{

//! @addtogroup cvv
//! @{

namespace impl
{
CV_EXPORTS void finalShow();
}

/** @brief Passes the control to the debug-window for a last time.

This function **must** be called *once* *after* all cvv calls if any. As an alternative create an
instance of FinalShowCaller, which calls finalShow() in its destructor (RAII-style).
 */
inline void finalShow()
{
#ifdef CVVISUAL_DEBUGMODE
	if (debugMode())
	{
		impl::finalShow();
	}
#endif
}

/**
 * @brief RAII-class to call finalShow() in it's dtor.
 */
class FinalShowCaller
{
public:
	/**
	 * @brief Calls finalShow().
	 */
	~FinalShowCaller()
	{
		finalShow();
	}
};

//! @}

}

#endif
