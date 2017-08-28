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

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <cstdlib>

int main(int argc, char **argv)
{

    if (argc < 4)
    {
      std::cerr << "Usage " << argv[0] << ": "
                << "<model-definition-file> " << " "
                << "<model-weights-file> " << " "
                << "<test-image>\n";
      return -1;

    }
    cv::String model_prototxt = argv[1];
    cv::String model_binary = argv[2];
    cv::String test_image = argv[3];
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(model_prototxt, model_binary);

    if (net.empty())
    {
        std::cerr << "Couldn't load the model !\n";
        return -2;
    }
    cv::Mat img = cv::imread(test_image);
    if (img.empty())
    {
        std::cerr << "Couldn't load image: " << test_image << "\n";
        return -3;
    }

    cv::Mat input_blob = cv::dnn::blobFromImage(
      img, 1.0, cv::Size(416, 416), cv::Scalar(104, 117, 123), false);

    cv::Mat prob;
    cv::TickMeter t;

    net.setInput(input_blob);
    t.start();
    prob = net.forward("predictions");
    t.stop();

    int prob_size[3] = {1000, 1, 1};
    cv::Mat prob_data(3, prob_size, CV_32F, prob.ptr<float>(0));

    double max_prob = -1.0;
    int class_idx = -1;
    for (int idx = 0; idx < prob.size[1]; ++idx)
    {
        double current_prob = prob_data.at<float>(idx, 0, 0);
        if (current_prob > max_prob)
        {
          max_prob = current_prob;
          class_idx = idx;
        }
    }
    std::cout << "Best class Index: " << class_idx << "\n";
    std::cout << "Time taken: " << t.getTimeSec() << "\n";
    std::cout << "Probability: " << max_prob * 100.0<< "\n";

    return 0;
}
