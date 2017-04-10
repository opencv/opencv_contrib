/*
  By downloading, copying, installing or using the software you agree to this license.
  If you do not agree to this license, do not download, install,
  copy or use the software.


                            License Agreement
                 For Open Source Computer Vision Library
                         (3-clause BSD License)

  Copyright (C) 2000-2016, Intel Corporation, all rights reserved.
  Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
  Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
  Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
  Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
  Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
  Third party copyrights are property of their respective owners.

  Redistribution and use in source and binary forms, with or without modification,
  are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the names of the copyright holders nor the names of the contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

  This software is provided by the copyright holders and contributors "as is" and
  any express or implied warranties, including, but not limited to, the implied
  warranties of merchantability and fitness for a particular purpose are disclaimed.
  In no event shall copyright holders or contributors be liable for any direct,
  indirect, incidental, special, exemplary, or consequential damages
  (including, but not limited to, procurement of substitute goods or services;
  loss of use, data, or profits; or business interruption) however caused
  and on any theory of liability, whether in contract, strict liability,
  or tort (including negligence or otherwise) arising in any way out of
  the use of this software, even if advised of the possibility of such damage.
 */

#ifndef __OPENCV_DNN_M_HPP__
#define __OPENCV_DNN_M_HPP__

#include "opencv2/core.hpp"

/** @defgroup dnn_modern Deep Learning Modern Module
 * This module is based on the [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn) framework.
 * The module uses tiny-dnn to load and run pre-trained Caffe models.
 * tiny-dnn's converter only supports single input/single output network without branches.
*/


namespace cv {
namespace dnn2 {

//! @addtogroup dnn_modern
//! @{

/** @brief Base class for tiny-dnn converter.
 */
class CV_EXPORTS_W BaseConverter
{
public:
    virtual ~BaseConverter() {};

    /**
    @brief Evaluates single model output on single model input.
    @param image input image.
    @param results output form model.
    */
    CV_WRAP virtual void eval(InputArray image, std::vector<float>& results) = 0;
};

/** @brief Class implementing the CaffeConverter.

Implementation of tiny-dnn Caffe converter.
Loads a pretrained Caffe model. Only support simple sequential models.

 */
class CV_EXPORTS_W CaffeConverter : public BaseConverter {
 public:

    /**
    @brief Creates a CaffeConverter object.

    @param model_file path to the prototxt file.
    @param trained_file path to the caffemodel file.
    @param mean_file path to binaryproto file.
    */
    CV_WRAP static Ptr<CaffeConverter> create(const String& model_file,
                                              const String& trained_file,
                                              const String& mean_file = String());

    CV_WRAP virtual void eval(InputArray image, CV_OUT std::vector<float>& results) = 0;
};

//! @}
} // namespace dnn2
} // namespace cv
#endif

/* End of file. */
