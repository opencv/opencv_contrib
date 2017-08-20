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

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace cv;
using namespace std;
using namespace cv::dnn;


int main(int argc, char **argv) {
    std::string base_dir = "/home/kv/gsoc/practice/";
    std::string model_prototxt = "/home/kv/gsoc/opencv_contrib/modules/dnn_objdetect/proto/SqueezeDet_deploy.prototxt";
    std::string model_binary = base_dir + "snapshot_iter_160000.caffemodel";
    std::string test_input_image = (argc > 1) ? argv[1] : "space_shuttle.jpg";

    std::cout << "Loading the network...\n";
    Net net = dnn::readNetFromCaffe(model_prototxt, model_binary);
    if (net.empty()) {
       std::cerr << "Couldn't load the model !\n";
       return -1;
    } else {
      std::cout << "Done loading the network !\n\n";
    }

    Mat img = cv::imread(test_input_image, CV_LOAD_IMAGE_COLOR);
    if (img.empty()) {
        std::cerr << "Couldn't load image: " << test_input_image << "\n";
        return -2;
    }

    resize(img, img, Size(416, 416));
    Mat inputBlob = blobFromImage(img, 1.0, Size(), Scalar(), false);
    net.setInput(inputBlob);

    std::cout << "Getting the output of all the three blobs...\n";
    std::vector<Mat> outblobs;
    std::vector<cv::String> out_layers;
    out_layers.push_back("slice");
    out_layers.push_back("softmax");
    out_layers.push_back("sigmoid");
    std::vector<cv::String> out_blobs;
    out_blobs.push_back("delta_bbox");
    out_blobs.push_back("class_scores");
    out_blobs.push_back("conf_scores");

    // Make this more efficient, couldn't find an alternate to this
    for (size_t layer = 0; layer < out_layers.size(); ++layer) {
        std::vector<Mat> temp_blob;
        net.forward(temp_blob, out_layers[layer]);
        if (layer == 0) {
            outblobs.push_back(temp_blob[2]);
        } else {
            outblobs.push_back(temp_blob[0]);
        }
    }
    std::cout << "Done !\n";

    // Check that the blobs are valid
    for (size_t i = 0; i < outblobs.size(); ++i) {
        if (outblobs[i].empty()) {
            std::cerr << "Blob: " << i << " is empty !\n";
        }
    }
    
    int delta_bbox_size[3] = {23, 23, 36};
    Mat delta_bbox(3, delta_bbox_size, CV_32F, outblobs[0].ptr<float>());

    int class_scores_size[2] = {4761, 20};
    Mat class_scores(2, class_scores_size, CV_32F, outblobs[1].ptr<float>());

    int conf_scores_size[3] = {23, 23, 9};
    Mat conf_scores(3, conf_scores_size, CV_32F, outblobs[2].ptr<float>());

    return 0;
}
