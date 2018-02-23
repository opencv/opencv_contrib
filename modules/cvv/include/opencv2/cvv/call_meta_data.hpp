#ifndef CVVISUAL_CALL_DATA_HPP
#define CVVISUAL_CALL_DATA_HPP

#include <string>
#include <cstddef>
#include <utility>

namespace cvv
{

//! @addtogroup cvv
//! @{

namespace impl
{

/**
 * @brief Optional information about a location in Code.
 */
struct CallMetaData
{
      public:
	/**
	 * @brief Creates an unknown location.
	 */
	CallMetaData()
	    : file(nullptr), line(0), function(nullptr), isKnown(false)
	{
	}

	/**
	 * @brief Creates the provided location.
	 *
	 * Argument should be self-explaining.
	 */
	CallMetaData(const char *file, size_t line, const char *function)
	    : file(file), line(line), function(function), isKnown(true)
	{
	}
	operator bool()
	{
		return isKnown;
	}

	// self-explaining:
	const char *file;
	const size_t line;
	const char *function;

	/**
	 * @brief Whether *this holds actual data.
	 */
	const bool isKnown;
};
}

//! @}

} // namespaces

/**
 * @brief Creates an instance of CallMetaData with the location of the macro as
 * value.
 */
#define CVVISUAL_LOCATION ::cvv::impl::CallMetaData(__FILE__, __LINE__, CV_Func)

#endif
