// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>
#include <sstream>
#include <opencv2/dnn_superres.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn_superres;

int main(int argc, char *argv[])
{
    // Check for valid command line arguments, print usage
    // if insufficient arguments were given.
    if (argc < 4) {
        cout << "usage:   Arg 1: image     | Path to image" << endl;
        cout << "\t Arg 2: scales in a format of 2,4,8\n";
        cout << "\t Arg 3: output node names in a format of nchw_output_0,nchw_output_1\n";
        cout << "\t Arg 4: path to model file \n";
        return -1;
    }

    string img_path = string(argv[1]);
    string scales_str = string(argv[2]);
    string output_names_str = string(argv[3]);
    std::string path = string(argv[4]);

    //Parse the scaling factors
    std::vector<int> scales;
    char delim = ',';
    {
        std::stringstream ss(scales_str);
        std::string token;
        while (std::getline(ss, token, delim)) {
            scales.push_back(atoi(token.c_str()));
        }
    }

    //Parse the output node names
    std::vector<String> node_names;
    {
        std::stringstream ss(output_names_str);
        std::string token;
        while (std::getline(ss, token, delim)) {
            node_names.push_back(token);
        }
    }

    // Load the image
    Mat img = cv::imread(img_path);
    Mat original_img(img);
    if (img.empty())
    {
        std::cerr << "Couldn't load image: " << img << "\n";
        return -2;
    }

    //Make dnn super resolution instance
    DnnSuperResImpl sr;
    int scale = *max_element(scales.begin(), scales.end());
    std::vector<Mat> outputs;
    sr.readModel(path);
    sr.setModel("lapsrn", scale);

    sr.upsampleMultioutput(img, outputs, scales, node_names);

    for(unsigned int i = 0; i < outputs.size(); i++)
    {
        cv::namedWindow("Upsampled image", WINDOW_AUTOSIZE);
        cv::imshow("Upsampled image", outputs[i]);
        //cv::imwrite("./saved.jpg", img_new);
        cv::waitKey(0);
    }

    return 0;
}