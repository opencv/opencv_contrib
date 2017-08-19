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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
        InferBbox(Mat delta_bbox, Mat class_scores, Mat conf_scores,
                  size_t n_top_detections=64, double intersection_thresh=0.8);
        std::vector<object> filter();
        void transform_bboxes(std::vector<std::vector<double> > *bboxes);
        void final_probability_dist(std::vector<std::vector<double> > *final_probs);
        void transform_bboxes_inv(std::vector<std::vector<double> > *pre,
                                  std::vector<std::vector<double> > *post);
        void assert_predictions(std::vector<std::vector<double> > *min_max_boxes);
        void non_maximal_supression();
      private:
        Mat delta_bbox;
        Mat class_scores;
        Mat conf_scores;

        // Image Dimensions
        size_t image_width;
        size_t image_height;

        // ConvDet feature map details
        size_t W, H;
        std::vector<std::vector<double> > anchors_values;
        std::vector<std::pair<double, double> > anchor_center;
        std::vector<std::pair<double, double> > anchor_shapes;

        std::map<int, std::string> label_map;

        size_t num_classes;
        size_t anchors_per_grid;
        size_t anchors;
        double intersection_thresh;
        size_t n_top_detections;
    };

  }  //  namespace dnn_objdetect

}  //  namespace cv

#endif
