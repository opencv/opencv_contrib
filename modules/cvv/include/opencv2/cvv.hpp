#ifndef __OPENCV_CVV_HPP__
#define __OPENCV_CVV_HPP__

/**
@defgroup cvv GUI for Interactive Visual Debugging of Computer Vision Programs

Namespace for all functions is **cvv**, i.e. *cvv::showImage()*.

Compilation:

-   For development, i.e. for cvv GUI to show up, compile your code using cvv with
    *g++ -DCVVISUAL_DEBUGMODE*.
-   For release, i.e. cvv calls doing nothing, compile your code without above flag.

See cvv tutorial for a commented example application using cvv.

*/

#include <opencv2/cvv/call_meta_data.hpp>
#include <opencv2/cvv/debug_mode.hpp>
#include <opencv2/cvv/dmatch.hpp>
#include <opencv2/cvv/filter.hpp>
#include <opencv2/cvv/final_show.hpp>
#include <opencv2/cvv/show_image.hpp>

#endif //__OPENCV_CVV_HPP__
