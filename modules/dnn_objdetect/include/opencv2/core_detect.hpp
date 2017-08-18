/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.
                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)
Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
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

#ifndef _OPENCV_DNN_OBJDETECT_CORE_DETECT_HPP_
#define _OPENCV_DNN_OBJDETECT_CORE_DETECT_HPP_

#include <vector>
#include <memory>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

namespace cv {
  namespace dnn_objdetect {

    typedef struct {
      int xmin, xmax;
      int ymin, ymax;
      int class_idx;
    } object;

    /** @brief This class takes in the network output and predicts the bbox(s)
     *
     */
    class CV_EXPORTS InferBbox {
      public:
        InferBbox(Mat conf_scores, Mat bbox_delta, Mat class_scores);
        std::vector<object> filter();
        void non_maximal_supression();
      private:
        Mat conf_scores;
        Mat bbox_delta;
        Mat class_scores;

        // Image Dimensions
        size_t image_width;
        size_t image_height;

        // ConvDet feature map details
        size_t W, H;
        std::vector<std::vector<double> > anchors_values;
        std::vector<std::pair<double, double> > anchor_center;
        std::vector<std::pair<double, double> > anchor_shapes;

        size_t num_classes;
        size_t anchors_per_grid;
        size_t anchors;
        double intersection_thresh;
        size_t n_top_detections;
    };

  }  //  namespace dnn_objdetect

}  //  namespace cv





#endif
