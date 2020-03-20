// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>
#include "opencv2/highgui.hpp"
#include <opencv2/core/base.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/infoflow.hpp>

using namespace std;
using namespace cv;
using namespace cv::alphamat;

const char* keys =
{
    "{img || input image name}"
    "{tri || input trimap image name}"
    "{out || output image name}"
};

int main(int argc, char *argv[])
{
    bool show_help = (argc == 1);
    show_help = show_help || (argc == 2 && string(argv[1]) == "--help");
    show_help = show_help || (argc == 2 && string(argv[1]) == "-h");

    if (show_help)
    {
        printf("\nThis sample demonstrates Information Flow alpha matting\n"
               "Call:\n"
               "    alphamat_information_flow -img=<string> -tri=<string> [-out=<string>]\n\n");
        return 0;
    }

    CommandLineParser parser(argc, argv, keys);
    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    string img_path = parser.get<std::string>("img");
    string trimap_path = parser.get<std::string>("tri");
    string result_path = parser.get<std::string>("out");

    Mat image, tmap;

    image = imread(img_path, IMREAD_COLOR);   // Read the input image file
    if (image.empty())
    {
        printf("Cannot read image file: %s\n", img_path.c_str());
        return -1;
    }

    tmap = imread(trimap_path, IMREAD_GRAYSCALE);
    if (tmap.empty())
    {
        printf("Cannot read trimap file: %s\n", trimap_path.c_str());
        return -1;
    }

    Mat result;
    infoFlow(image, tmap, result, false, true);

    if (result_path.empty())
    {
        namedWindow("result alpha matte", WINDOW_NORMAL);
        imshow("result alpha matte", result);
        waitKey(0);
    }
    else
    {
        imwrite(result_path, result);
    }

    imshow("Result Matte", result);
    return 0;

}
