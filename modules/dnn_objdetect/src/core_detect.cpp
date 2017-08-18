namespace cv
{
  namespace dnn_objdetect
  {
    InferBbox::InferBbox(Mat conf_scores, Mat bbox_delta, Mat class_scores)
    {
      this->conf_scores = conf_scores;
      this->bbox_delta = bbox_delta;
      this->class_scores = class_scores;

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
    }

  }  //  namespace cv

}  //  namespace dnn_objdetect
