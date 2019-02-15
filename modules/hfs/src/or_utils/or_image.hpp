// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_OR_IMAGE_HPP_
#define _OPENCV_OR_IMAGE_HPP_


#include "or_memory_block.hpp"
namespace cv { namespace hfs { namespace orutils {

template <typename T>
class Image : public MemoryBlock < T >
{
public:
    Vector2<int> noDims;

    Image( Vector2<int> noDims_ )
        : MemoryBlock<T>( noDims_.x * noDims_.y )
    {
        this->noDims = noDims_;
    }

    Image(const Image&);
    Image& operator=(const Image&);
};

}}}

#endif
