// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/ovis.hpp"
#include "opencv2/opencv_modules.hpp"

#include <Ogre.h>

namespace cv {
namespace ovis {
struct Application;
extern Ptr<Application> _app;

extern const char* RESOURCEGROUP_NAME;
void _createTexture(const String& name, Mat image);
}
}

#endif
