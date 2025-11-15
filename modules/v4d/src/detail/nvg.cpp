// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/nvg.hpp"
#include <nanovg.h>
#include "opencv2/core.hpp"

#include <cstring>

namespace cv {
namespace v4d {
/*!
 * In general please refere to https://github.com/memononen/nanovg/blob/master/src/nanovg.h for reference.
 */
namespace nvg {
Paint::Paint(const NVGpaint& np) {
    memcpy(this->xform, np.xform, sizeof(this->xform));
    memcpy(this->extent, np.extent, sizeof(this->extent));
    this->radius = np.radius;
    this->feather = np.feather;
    this->innerColor = cv::Scalar(np.innerColor.rgba[2] * 255, np.innerColor.rgba[1] * 255,
            np.innerColor.rgba[0] * 255, np.innerColor.rgba[3] * 255);
    this->outerColor = cv::Scalar(np.outerColor.rgba[2] * 255, np.outerColor.rgba[1] * 255,
            np.outerColor.rgba[0] * 255, np.outerColor.rgba[3] * 255);
    this->image = np.image;
}

NVGpaint Paint::toNVGpaint() {
    NVGpaint np;
    memcpy(np.xform, this->xform, sizeof(this->xform));
    memcpy(np.extent, this->extent, sizeof(this->extent));
    np.radius = this->radius;
    np.feather = this->feather;
    np.innerColor = nvgRGBA(this->innerColor[2], this->innerColor[1], this->innerColor[0],
            this->innerColor[3]);
    np.outerColor = nvgRGBA(this->outerColor[2], this->outerColor[1], this->outerColor[0],
            this->outerColor[3]);
    np.image = this->image;
    return np;
}
}
}
}
