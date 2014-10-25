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
 // Copyright (C) 2014, OpenCV Foundation, all rights reserved.
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

#include "precomp.hpp"
#include "opencv2/saliency.hpp"

namespace cv
{
namespace saliency
{

CV_INIT_ALGORITHM( StaticSaliencySpectralResidual, "SALIENCY.SPECTRAL_RESIDUAL",
                   obj.info()->addParam( obj, "resImWidth", obj.resImWidth); obj.info()->addParam( obj, "resImHeight", obj.resImHeight) );


CV_INIT_ALGORITHM(
    ObjectnessBING, "SALIENCY.BING",
    obj.info()->addParam(obj, "_base", obj._base); obj.info()->addParam(obj, "_NSS", obj._NSS); obj.info()->addParam(obj, "_W", obj._W) );

CV_INIT_ALGORITHM( MotionSaliencyBinWangApr2014, "SALIENCY.BinWangApr2014",
                   obj.info()->addParam( obj, "imageWidth", obj.imageWidth); obj.info()->addParam( obj, "imageHeight", obj.imageHeight) );

bool initModule_saliency( void )
{
  bool all = true;
  all &= !StaticSaliencySpectralResidual_info_auto.name().empty();
  //all &= !MotionSaliencySuBSENSE_info_auto.name().empty();
  all &= !MotionSaliencyBinWangApr2014_info_auto.name().empty();
  all &= !ObjectnessBING_info_auto.name().empty();

  return all;
}

}  // namespace saliency
}  // namespace cv
