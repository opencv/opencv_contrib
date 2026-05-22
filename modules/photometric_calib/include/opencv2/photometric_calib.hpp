// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_PHOTOMETRIC_CALIB_HPP__
#define __OPENCV_PHOTOMETRIC_CALIB_HPP__

#include "opencv2/photometric_calib/Reader.hpp"
#include "opencv2/photometric_calib/GammaRemover.hpp"
#include "opencv2/photometric_calib/VignetteRemover.hpp"
#include "opencv2/photometric_calib/ResponseCalib.hpp"
#include "opencv2/photometric_calib/VignetteCalib.hpp"

/**
 * @defgroup photometric_calib Photometric Calibration
 * The photometric_calib contains photomeric calibration algorithm proposed by Jakob Engel. \n
 * The implementation is totally based on the paper \cite engel2016monodataset. \n
 * Photometric calibration aimed at removing the camera response function and vitnetting artefact,
 * by which the tracking and image alignment algorithms based on direct methods can be improved significantly. \n
 * For details please refer to \cite engel2016monodataset.
 */

namespace cv {
namespace photometric_calib {
} // namespace photometric_calib
} // namespace cv

#endif