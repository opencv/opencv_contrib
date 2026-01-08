// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"

#include "opencv2/core_detect.hpp"


namespace cv
{
  namespace dnn_objdetect
  {
    InferBbox::InferBbox(Mat _delta_bbox, Mat _class_scores, Mat _conf_scores)
    {
      this->delta_bbox = _delta_bbox;
      this->class_scores = _class_scores;
      this->conf_scores = _conf_scores;

      image_width = 416;
      image_height = 416;

      W = 23;
      H = 23;
      num_classes = 20;
      anchors_per_grid = 9;
      anchors = W * H * anchors_per_grid;

      intersection_thresh = 0.65;
      nms_intersection_thresh = 0.1;
      n_top_detections = 64;
      epsilon = 1e-7;

      anchors_values.resize(anchors);
      for (size_t i = 0; i < anchors; ++i)
      {
        anchors_values[i].resize(4);
      }

      // Anchor shapes predicted from kmeans clustering
      double arr[9][2] = {{377, 371}, {64, 118}, {129, 326},
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
      for (size_t i = 0, anchor = 0, j = 0; anchor < anchors; ++anchor)
      {
        anchors_values[anchor][0] = anchor_center.at(i).first;
        anchors_values[anchor][1] = anchor_center.at(i).second;
        anchors_values[anchor][2] = anchor_shapes.at(j).first;
        anchors_values[anchor][3] = anchor_shapes.at(j).second;
        if ((anchor+1) % anchors_per_grid == 0)
        {
          i += 1;
          j = 0;
        }
        else
        {
          ++j;
        }
      }

      // Map the class index to the corresponding labels
      std::string arrs[20] = {"aeroplane", "bicycle", "bird", "boat",
                           "bottle", "bus", "car", "cat", "chair",
                           "cow", "diningtable", "dog", "horse",
                           "motorbike", "person", "pottedplant",
                           "sheep", "sofa", "train", "tvmonitor"};
      for (size_t idx = 0; idx < num_classes; ++idx)
      {
        label_map.push_back(arrs[idx]);
      }
    }

    void InferBbox::filter(double thresh)
    {
      this->intersection_thresh = thresh;
      // Some containers
      std::vector<std::vector<double> > transformed_bbox_preds(this->anchors);
      std::vector<std::vector<double> > min_max_bboxes(this->anchors);
      std::vector<std::vector<double> > final_probs(this->anchors);

      for (size_t i = 0; i < this->anchors; ++i)
      {
        transformed_bbox_preds[i].resize(4);
        final_probs[i].resize(num_classes);
        min_max_bboxes[i].resize(4);
      }

      // Transform relative coordinates from ConvDet to bounding box coordinates
      transform_bboxes(&transformed_bbox_preds);

      // Do the inverse transformation of the predicted bboxes
      transform_bboxes_inv(&transformed_bbox_preds, &min_max_bboxes);

      // Ensure that the predicted bounding boxes are well within the image
      // dimensions
      assert_predictions(&min_max_bboxes);

      // Compute the final probability values
      final_probability_dist(&final_probs);

      // Filter the classes of n_top_detections
      std::vector<std::vector<double> > top_n_boxes(n_top_detections);
      std::vector<size_t> top_n_idxs(n_top_detections);
      std::vector<double> top_n_probs(n_top_detections);
      for (size_t i = 0; i < n_top_detections; ++i)
      {
        top_n_boxes[i].resize(4);
      }

      filter_top_n(&final_probs, &min_max_bboxes, top_n_boxes,
        top_n_idxs, top_n_probs);

      // Apply Non-Maximal-Supression to the n_top_detections
      nms_wrapper(top_n_boxes, top_n_idxs, top_n_probs);

    }

    void InferBbox::transform_bboxes(std::vector<std::vector<double> > *bboxes)
    {
      for (unsigned int h = 0; h < H; ++h)
      {
        for (unsigned int w = 0; w < W; ++w)
        {
          for (unsigned int anchor = 0; anchor < anchors_per_grid; ++anchor)
          {
            const int anchor_idx = (h * W + w) * anchors_per_grid + anchor;
            double delta_x = this->delta_bbox.at<float>(h, w, anchor * 4 + 0);
            double delta_y = this->delta_bbox.at<float>(h, w, anchor * 4 + 1);
            double delta_h = this->delta_bbox.at<float>(h, w, anchor * 4 + 2);
            double delta_w = this->delta_bbox.at<float>(h, w, anchor * 4 + 3);

            (*bboxes)[anchor_idx][0] = this->anchors_values[anchor_idx][0] +
                          this->anchors_values[anchor_idx][3] * delta_x;
            (*bboxes)[anchor_idx][1] = this->anchors_values[anchor_idx][1] +
                          this->anchors_values[anchor_idx][2] * delta_y;;
            (*bboxes)[anchor_idx][2] =
                          this->anchors_values[anchor_idx][2] * exp(delta_h);
            (*bboxes)[anchor_idx][3] =
                          this->anchors_values[anchor_idx][3] * exp(delta_w);
          }
        }
      }
    }

    void InferBbox::final_probability_dist(
        std::vector<std::vector<double> > *final_probs)
    {
      for (unsigned int h = 0; h < H; ++h)
      {
        for (unsigned int w = 0; w < W; ++w)
        {
          for (unsigned int ch = 0; ch < anchors_per_grid * num_classes; ++ch)
          {
            const int anchor_idx =
              (h * W + w) * anchors_per_grid + ch / num_classes;
            double pr_object =
              conf_scores.at<float>(h, w, ch / num_classes);
            double pr_class_idx =
              class_scores.at<float>(anchor_idx, ch % num_classes);
            (*final_probs)[anchor_idx][ch % num_classes] =
              pr_object * pr_class_idx;
          }
        }
      }
    }

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
    }

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
    }

    void InferBbox::filter_top_n(std::vector<std::vector<double> >
      *probs, std::vector<std::vector<double> > *boxes,
      std::vector<std::vector<double> > &top_n_boxes,
      std::vector<size_t> &top_n_idxs,
      std::vector<double> &top_n_probs)
    {
      std::vector<double> max_class_probs((*probs).size());
      std::vector<size_t> args((*probs).size());

      for (unsigned int box = 0; box < (*boxes).size(); ++box)
      {
        size_t _prob_idx =
            std::max_element((*probs)[box].begin(),
            (*probs)[box].end()) - (*probs)[box].begin();
        max_class_probs[box] = (*probs)[box][_prob_idx];
      }

      std::vector<std::pair<double, size_t> > temp_sort(max_class_probs.size());
      for (size_t tidx = 0; tidx < max_class_probs.size(); ++tidx)
      {
        temp_sort[tidx] = std::make_pair(max_class_probs[tidx],
          static_cast<size_t>(tidx));
      }
      std::sort(temp_sort.begin(), temp_sort.end(), InferBbox::comparator);

      for (size_t idx = 0; idx < temp_sort.size(); ++idx)
      {
        args[idx] = temp_sort[idx].second;
      }

      // Get n_top_detections
      std::vector<size_t> top_n_order(args.begin(),
        args.begin() + n_top_detections);

      // Have a separate copy of all the n_top_detections
      for (size_t n = 0; n < n_top_detections; ++n)
      {
        top_n_probs[n] = max_class_probs[top_n_order[n]];
        top_n_idxs[n]  =
            std::max_element((*probs)[top_n_order[n]].begin(),
            (*probs)[top_n_order[n]].end()) -
            (*probs)[top_n_order[n]].begin();
        for (size_t i = 0; i < 4; ++i)
        {
          top_n_boxes[n][i] = (*boxes)[top_n_order[n]][i];
        }
      }
    }

    void InferBbox::nms_wrapper(std::vector<std::vector<double> >
      &top_n_boxes, std::vector<size_t> &top_n_idxs,
      std::vector<double> &top_n_probs)
    {
      for (size_t c = 0; c < this->num_classes; ++c)
      {
        std::vector<size_t> idxs_per_class;
        for (size_t n = 0; n < n_top_detections; ++n)
        {
          if (top_n_idxs[n] == c)
          {
            idxs_per_class.push_back(n);
          }
        }

        // Just continue in case there are no objects of this class
        if (idxs_per_class.size() == 0)
        {
          continue;
        }

        // Process per class detections
        std::vector<std::vector<double> > boxes_per_class(idxs_per_class.size());
        std::vector<double> probs_per_class(idxs_per_class.size());
        std::vector<bool> keep_per_class;
        for (std::vector<size_t>::iterator itr = idxs_per_class.begin();
            itr != idxs_per_class.end(); ++itr)
        {
          size_t idx = itr - idxs_per_class.begin();
          probs_per_class[idx] = top_n_probs[*itr];
          for (size_t b = 0; b < 4; ++b)
          {
            boxes_per_class[idx].push_back(top_n_boxes[*itr][b]);
          }
        }
        keep_per_class =
            non_maximal_suppression(&boxes_per_class, &probs_per_class);
        for (std::vector<bool>::iterator itr = keep_per_class.begin();
            itr != keep_per_class.end(); ++itr)
        {
          size_t idx = itr - keep_per_class.begin();
          if (*itr && probs_per_class[idx] > this->intersection_thresh)
          {
            dnn_objdetect::object new_detection;

            new_detection.class_idx = c;
            new_detection.label_name = this->label_map[c];
            new_detection.xmin = (int)boxes_per_class[idx][0];
            new_detection.ymin = (int)boxes_per_class[idx][1];
            new_detection.xmax = (int)boxes_per_class[idx][2];
            new_detection.ymax = (int)boxes_per_class[idx][3];
            new_detection.class_prob = probs_per_class[idx];

            this->detections.push_back(new_detection);
          }
        }
      }
    }

    std::vector<bool> InferBbox::non_maximal_suppression(
      std::vector<std::vector<double> > *boxes, std::vector<double>
      *probs)
    {
      std::vector<bool> keep(((*probs).size()));
      std::fill(keep.begin(), keep.end(), true);
      std::vector<size_t> prob_args_sorted((*probs).size());

      std::vector<std::pair<double, size_t> > temp_sort((*probs).size());
      for (size_t tidx = 0; tidx < (*probs).size(); ++tidx)
      {
        temp_sort[tidx] = std::make_pair((*probs)[tidx],
          static_cast<size_t>(tidx));
      }
      std::sort(temp_sort.begin(), temp_sort.end(), InferBbox::comparator);

      for (size_t idx = 0; idx < temp_sort.size(); ++idx)
      {
        prob_args_sorted[idx] = temp_sort[idx].second;
      }

      for (std::vector<size_t>::iterator itr = prob_args_sorted.begin();
          itr != prob_args_sorted.end()-1; ++itr)
      {
        size_t idx = itr - prob_args_sorted.begin();
        std::vector<double> iou_(prob_args_sorted.size() - idx - 1);
        std::vector<std::vector<double> > temp_boxes(iou_.size());
        for (size_t bb = 0; bb < temp_boxes.size(); ++bb)
        {
          std::vector<double> temp_box(4);
          for (size_t b = 0; b < 4; ++b)
          {
            temp_box[b] = (*boxes)[prob_args_sorted[idx + bb + 1]][b];
          }
          temp_boxes[bb] = temp_box;
        }
        intersection_over_union(&temp_boxes,
            &(*boxes)[prob_args_sorted[idx]], &iou_);
        for (std::vector<double>::iterator _itr = iou_.begin();
            _itr != iou_.end(); ++_itr)
        {
          size_t iou_idx = _itr - iou_.begin();
          if (*_itr > nms_intersection_thresh)
          {
            keep[prob_args_sorted[idx+iou_idx+1]] = false;
          }
        }
      }
      return keep;
    }

    void InferBbox::intersection_over_union(std::vector<std::vector<double> >
      *boxes, std::vector<double> *base_box, std::vector<double> *iou)
    {
      double g_xmin = (*base_box)[0];
      double g_ymin = (*base_box)[1];
      double g_xmax = (*base_box)[2];
      double g_ymax = (*base_box)[3];
      double base_box_w = g_xmax - g_xmin;
      double base_box_h = g_ymax - g_ymin;
      for (size_t b = 0; b < (*boxes).size(); ++b)
      {
        double xmin = std::max((*boxes)[b][0], g_xmin);
        double ymin = std::max((*boxes)[b][1], g_ymin);
        double xmax = std::min((*boxes)[b][2], g_xmax);
        double ymax = std::min((*boxes)[b][3], g_ymax);

        // Intersection
        double w = std::max(static_cast<double>(0.0), xmax - xmin);
        double h = std::max(static_cast<double>(0.0), ymax - ymin);
        // Union
        double test_box_w = (*boxes)[b][2] - (*boxes)[b][0];
        double test_box_h = (*boxes)[b][3] - (*boxes)[b][1];

        double inter_ = w * h;
        double union_ = test_box_h * test_box_w + base_box_h * base_box_w - inter_;
        (*iou)[b] = inter_ / (union_ + epsilon);
      }
    }

  }

}
