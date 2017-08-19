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

#include "precomp.hpp"

#include <vector>
#include <memory>
#include <string>

#include "core_detect.hpp"


namespace cv
{
  namespace dnn_objdetect
  {
    InferBbox::InferBbox(Mat delta_bbox, Mat class_scores, Mat conf_scores,
      size_t n_top_detections = 64, double intersection_thresh = 0.8)
    {
      this->delta_bbox = bbox_delta;
      this->class_scores = class_scores;
      this->conf_scores = conf_scores;

      this->n_top_detections = n_top_detections;
      this->intersection_thresh = intersection_thresh;

      image_width = 416;
      image_height = 416;

      W = 23;
      H = 23;
      num_classes = 20;
      anchors_per_grid = 9;
      anchors = W * H * anchors_per_grid;

      anchors_values.resize(anchors);
      for (size_t i = 0; i < anchors; ++anchor)
      {
        anchors_values[i].resize(4);
      }

      // Anchor shapes predicted from kmeans clustering
      double arr[9][2] = {{377, 371}, {64, 118}, {129, 326}
                          {172, 126}, {34, 46}, {353, 204},
                          {89, 214}, {249, 361}, {209, 239}};
      for (size_t i = 0; i < anchors_per_grid; ++i)
      {
        anchor_shapes.push_back(std::make_pair(arr[i][1], arr[i][0]));
      }
      // Generate the anchor centers
      for (size_t x = 1; x < W + 1; ++x) {
        double c_x = (x * static_cast<double>(image_width)) / (W+1.0);
        for (size_t y = 1; y < H + 1; ++y) {
          double c_y = (y * static_cast<double>(image_height)) / (H+1.0);
          anchor_center.push_back(std::make_pair(c_x, c_y));
        }
      }

      // Generate the final anchor values
      for (size_t i = 0, anchor = 0, j = 0; anchor < anchors; ++anchor) {
        anchors_values[anchor][0] = anchor_center.at(i).first;
        anchors_values[anchor][1] = anchor_center.at(i).second;
        anchors_values[anchor][2] = anchor_shapes.at(j).first;
        anchors_values[anchor][3] = anchor_shapes.at(j).second;
        if ((anchor+1) % anchors_per_grid == 0) {
          i += 1;
          j = 0;
        } else {
          ++j;
        }
      }

      // Map the class index to the corresponding labels
      label_map = {0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 
                   4: "bottle", 5: "bus", 6: "car", 7: "cat", 8: "chair",
                   9: "cow", 10: "diningtable", 11: "dog", 12: "horse",
                   13: "motorbike", 14: "person", 15: "pottedplant", 
                   16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor"};
    }  //  default constructer

    std::vector<object> InferBbox::filter() 
    {
      // Some containers
      std::vector<std::vector<double> > transformed_bbox_preds;
      std::vector<std::vector<double> > min_max_bboxes;
      std::vector<std::vector<double> > final_probs;
      
      transformed_bbox_preds.resize(anchors);
      final_probs.resize(anchors);
      min_max_bboxes.resize(anchors);
      for (size_t i = 0; i < anchors; ++i) {
        transformed_bbox_preds[i].resize(4);
        final_probs[i].resize(num_classes);
        min_max_bboxes[i].resize(4);
      }
      
      // Transform relative coordinates from ConvDet to bounding box coordinates
      // @f$ [x_{i}^{p}, y_{j}^{p}, h_{k}^{p}, w_{k}^{p}] @f$
      transform_bboxes(&transformed_bbox_preds);

      // Do the inverse transformation of the predicted bboxes
      // from
      // @f$ [x_{i}^{p}, y_{j}^{p}, h_{k}^{p}, w_{k}^{p}] @f$
      // to
      // @f$ [xmin, ymin, xmax, ymax] @f$
      transform_bboxes_inv(&transformed_bbox_preds, &min_max_bboxes);

      // Ensure that the predicted bounding boxes are well within the image
      // dimensions
      assert_predictions(&min_max_bboxes);

      // Compute the final probability values
      final_probability_dist(&final_probs);

    }  //  filter function

    void InferBbox::transform_bboxes(std::vector<std::vector<double> > *bboxes)
    {
      for (size_t h = 0; h < H; ++h)
      {
        for (size_t w = 0; w < W; ++w)
        {
          for (size_t anchor = 0; anchor < anchors_per_grid; ++anchor)
          {
            const int anchor_idx = (h * W + w) * anchors_per_grid + anchor;
            double delta_x = this->delta_bbox.at<double>(h, w, anchor * 4 + 0);
            double delta_y = this->delta_bbox.at<double>(h, w, anchor * 4 + 1);
            double delta_h = this->delta_bbox.at<double>(h, w, anchor * 4 + 2);
            double delta_w = this->delta_bbox.at<double>(h, w, anchor * 4 + 3);

            (*bboxes)[anchor_idx][0] = this->anchors_values[anchor_idx][0] + \
                          this->anchors_values[anchor_idx][3] * delta_x;
            (*bboxes)[anchor_idx][1] = this->anchors_values[anchor_idx][1] + \
                          this->anchors_values[anchor_idx][2] * delta_y;;
            (*bboxes)[anchor_idx][2] = \
                          this->anchors_values[anchor_idx][2] * exp(delta_h);
            (*bboxes)[anchor_idx][3] = \
                          this->anchors_values[anchor_idx][3] * exp(delta_w);
          }
        }
      }
    }  //  transform_bboxes function

    void InferBbox::final_probability_dist(
        std::vector<std::vector<double> > *final_probs)
    {
      for (size_t h = 0; h < H; ++h)
      {
        for (size_t w = 0; w < W; ++w)
        {
          for (size_t ch = 0; ch < anchors_per_grid * num_classes; ++ch)
          {
            const int anchor_idx = \
              (h * W + w) * anchors_per_grid + ch / num_classes;
            double pr_object = \
              conf_scores.at<double>(h, w, ch / num_classes);
            double pr_class_idx = \
              class_scores.at<double>(anchor_idx, ch % num_classes);
            (*final_probs)[anchor_idx][ch % num_classes] = \
              pr_object * pr_class_idx;
          }
        }
      }
    }  // final_probability_dist function

    void InferBbox::transform_bboxes_inv(
      std::vector<std::vector<double> > *pre,
      std::vector<std::vector<double> > *post)
    {
      for (size_t anchor = 0; anchor < anchors; ++anchor)
      {
        double c_x = (*pre)[anchor][0];
        double c_y = (*pre)[anchor][1];
        double b_h = (*pre)[anchor][2];
        double b_w = (*pre)[anchor][3];

        (*post)[anchor][0] = c_x - b_w / 2.0;
        (*post)[anchor][1] = c_y - b_h / 2.0;
        (*post)[anchor][2] = c_x + b_w / 2.0;
        (*post)[anchor][3] = c_y + b_h / 2.0;
      }
    }  // transform_bboxes_inv function

    void InferBbox::assert_predictions(std::vector<std::vector<double> >
         *min_max_boxes)
    {
      for (size_t anchor = 0; anchor < anchors; ++anchor)
      {
        double p_xmin = (*min_max_boxes)[anchor][0];
        double p_ymin = (*min_max_boxes)[anchor][1];
        double p_xmax = (*min_max_boxes)[anchor][2];
        double p_ymax = (*min_max_boxes)[anchor][3];

        (*min_max_boxes)[anchor][0] = std::min(std::max(
          static_cast<double>(0.0), p_xmin), image_width -
          static_cast<double>(1.0));
        (*min_max_boxes)[anchor][1] = std::min(std::max(
          static_cast<double>(0.0), p_ymin), image_height -
          static_cast<double>(1.0));
        (*min_max_boxes)[anchor][2] = std::max(std::min(
          image_width - static_cast<double>(1.0), p_xmax),
          static_cast<double>(0.0));
        (*min_max_boxes)[anchor][3] = std::max(std::min(
          image_height - static_cast<double>(1.0), p_ymax),
          static_cast<double>(0.0));
      }
    }  //  assert_predictions function

  }  //  namespace cv

}  //  namespace dnn_objdetect
