/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#ifndef __OPENCV_SFM_IO_HPP__
#define __OPENCV_SFM_IO_HPP__

#include <opencv2/core.hpp>

namespace cv
{
namespace sfm
{

//! @addtogroup io
//! @{

/** @brief Different supported file formats.
 */
enum {
  SFM_IO_BUNDLER = 0,
  SFM_IO_VISUALSFM = 1,
  SFM_IO_OPENSFM = 2,
  SFM_IO_OPENMVG = 3,
  SFM_IO_THEIASFM = 4
};

/** @brief Import a reconstruction file.
  @param file The path to the file.
  @param Rs Output vector of 3x3 rotations of the camera
  @param Ts Output vector of 3x1 translations of the camera.
  @param Ks Output vector of 3x3 instrinsics of the camera.
  @param points3d Output array with 3d points. Is 3 x N.
  @param file_format The format of the file to import.

  The function supports reconstructions from Bundler.
*/
CV_EXPORTS_W
void
importReconstruction(const cv::String &file, OutputArrayOfArrays Rs,
                     OutputArrayOfArrays Ts, OutputArrayOfArrays Ks,
                     OutputArrayOfArrays points3d, int file_format = SFM_IO_BUNDLER);

//! @} sfm

} /* namespace sfm */
} /* namespace cv */

#endif

/* End of file. */
