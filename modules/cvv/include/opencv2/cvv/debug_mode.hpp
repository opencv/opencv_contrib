#ifndef CVVISUAL_DEBUG_MODE_HPP
#define CVVISUAL_DEBUG_MODE_HPP

#if __cplusplus >= 201103L && defined CVVISUAL_USE_THREAD_LOCAL
#define CVVISUAL_THREAD_LOCAL thread_local
#else
#define CVVISUAL_THREAD_LOCAL
#endif

namespace cvv
{

//! @addtogroup cvv
//! @{

namespace impl
{

/**
 * The debug-flag-singleton
 */
static inline bool &getDebugFlag()
{
	CVVISUAL_THREAD_LOCAL static bool flag = true;
	return flag;
}

} // namespace impl

/** @brief Returns whether debug-mode is active for this TU and thread.
*/
static inline bool debugMode()
{
	return impl::getDebugFlag();
}

/** @brief Enable or disable cvv for current translation unit and thread

(disabled this way has higher - but still low - overhead compared to using the compile flags).
@param active
 */
static inline void setDebugFlag(bool active)
{
	impl::getDebugFlag() = active;
}

//! @}

} // namespace cvv

#endif
