// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv {
namespace haze_removal{

HazeRemovalBase::HazeRemovalBase()
{
}

HazeRemovalBase::~HazeRemovalBase()
{
}

void HazeRemovalBase::dehaze(InputArray _src, OutputArray _dst)
{
    pImpl->dehaze(_src, _dst);
}

}} /// cv::haze_removal::
