// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>
#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/alphamat.hpp>

using namespace std;
using namespace cv;
using namespace cv::alphamat;

const char* keys =
    "{img || input image name}"
    "{tri || input trimap image name}"
    "{out || output image name}"
    "{help h || print help message}"
;

int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("This sample demonstrates Information Flow Alpha Matting");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string img_path = parser.get<std::string>("img");
    string trimap_path = parser.get<std::string>("tri");
    string result_path = parser.get<std::string>("out");

    if (!parser.check()
            || img_path.empty() || trimap_path.empty())
    {
        parser.printMessage();
        parser.printErrors();
        return 1;
    }

    Mat image, tmap;

    image = imread(img_path, IMREAD_COLOR);  // Read the input image file
    if (image.empty())
    {
        printf("Cannot read image file: '%s'\n", img_path.c_str());
        return 1;
    }

    tmap = imread(trimap_path, IMREAD_GRAYSCALE);
    if (tmap.empty())
    {
        printf("Cannot read trimap file: '%s'\n", trimap_path.c_str());
        return 1;
    }

    Mat result;
    infoFlow(image, tmap, result);

    if (result_path.empty())
    {
        namedWindow("result alpha matte", WINDOW_NORMAL);
        imshow("result alpha matte", result);
        waitKey(0);
    }
    else
    {
        imwrite(result_path, result);
        printf("Result saved: '%s'\n", result_path.c_str());
    }

    return 0;
}
