/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include <opencv2/opencv_modules.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/mcc.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>


using namespace std;
using namespace cv;
using namespace mcc;

const char *about = "Basic chart detection using neural network";
const char *keys = {
    "{t              |         | chartType: 0-Standard, 1-DigitalSG, 2-Vinyl}"
    "{m        |       | File path of model, if you don't have the model you can find the link in the documentation}"
    "{pb        |       | File path of pbtxt file, available along with with the model file }"
    "{v        |       | Input from video file, if ommited, input comes from camera }"
    "{ci       | 0     | Camera id if input doesnt come from video (-v) }"};

int main(int argc, char *argv[])
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if (argc < 4)
    {
        parser.printMessage();
        return 0;
    }

    CV_Assert(0<=parser.get<int>("t") && parser.get<int>("t")<3);
    TYPECHART chartType = TYPECHART(parser.get<int>("t"));
    string model_path = parser.get<string> ("m");
    string pbtxt_path = parser.get<string> ("pb");
    int camId = parser.get<int>("ci");

    String video;
    if (parser.has("v"))
    {
        video = parser.get<String>("v");
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    VideoCapture inputVideo;
    int waitTime;
    if (!video.empty())
    {
        inputVideo.open(video);
        waitTime = 10;
    }
    else
    {
        inputVideo.open(camId);
        waitTime = 10;
    }

    //load the network

	cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model_path, pbtxt_path);
    net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(dnn::DNN_TARGET_CUDA);

    Ptr<CCheckerDetector> detector = CCheckerDetector::create();
    if(!detector->setNet(net))
    {
        cout<<"Loading Model failed: Falling back to standard techniques"<<endl;
    }

    while (inputVideo.grab())
    {
        Mat image, imageCopy;
        inputVideo.retrieve(image);

        imageCopy=image.clone();
        cv::Rect region = Rect(Point2f(0,0), image.size());

        int max_number_of_charts_in_image = 2;

        // Marker type to detect
        if (!detector->process(image, chartType, max_number_of_charts_in_image, true))
        {
            printf("ChartColor not detected \n");
        }
        else
        {

            // get checker
            std::vector<Ptr<mcc::CChecker>> checkers;
            detector->getListColorChecker(checkers);
            for(Ptr<mcc::CChecker> checker: checkers)
            {
                // current checker

                Ptr<CCheckerDraw> cdraw = CCheckerDraw::create(checker);
                cdraw->draw(image);
            }

        }

        imshow("image result | q or esc to quit", image);
        imshow("original", imageCopy);
        char key = (char)waitKey(waitTime);
        if (key == 27)
            break;
    }

    return 0;
}
