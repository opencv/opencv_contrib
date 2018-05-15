// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE file found in this module's directory

#include "precomp.hpp"

namespace cv {
namespace kinfu {


#if PRINT_TIME

ScopeTime::ScopeTime(std::string name_, bool _enablePrint) :
    name(name_), enablePrint(_enablePrint)
{
    start = (double)cv::getTickCount();
    nested++;
}

ScopeTime::~ScopeTime()
{
    double time_ms =  ((double)cv::getTickCount() - start)*1000.0/cv::getTickFrequency();
    if(enablePrint)
    {
        std::string spaces(nested, '-');
        std::cout << spaces << "Time(" << name << ") = " << time_ms << " ms" << std::endl;
    }
    nested--;
}

int ScopeTime::nested  = 0;

#else

ScopeTime::ScopeTime(std::string /*name_*/, bool /*_enablePrint = true*/)
{ }

ScopeTime::~ScopeTime()
{ }

} // namespace kinfu
} // namespace cv
#endif
