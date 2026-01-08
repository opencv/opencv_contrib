// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Include relevant headers
#include <iostream>
#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/alphamat.hpp>

// Set namespaces
using namespace std;
using namespace cv;
using namespace cv::alphamat;

// Set the usage parameter names
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

    // Read the paths to the input image, input trimap and the location of the output image.
    string img_path = parser.get<std::string>("img");
    string trimap_path = parser.get<std::string>("tri");
    string result_path = parser.get<std::string>("out");

    // Make sure the user inputs paths to the input image and trimap
    if (!parser.check()
            || img_path.empty() || trimap_path.empty())
    {
        parser.printMessage();
        parser.printErrors();
        return 1;
    }

    Mat image, tmap;

    // Read the input image
    image = imread(img_path, IMREAD_COLOR);
    if (image.empty())
    {
        printf("Cannot read image file: '%s'\n", img_path.c_str());
        return 1;
    }

    // Read the trimap
    tmap = imread(trimap_path, IMREAD_GRAYSCALE);
    if (tmap.empty())
    {
        printf("Cannot read trimap file: '%s'\n", trimap_path.c_str());
        return 1;
    }

    Mat result;
    // Perform information flow alpha matting
    infoFlow(image, tmap, result);

    if (result_path.empty())
    {
        // Show the alpha matte if a result filepath is not provided.
        namedWindow("result alpha matte", WINDOW_NORMAL);
        imshow("result alpha matte", result);
        waitKey(0);
    }
    else
    {
        // Save the alphamatte
        imwrite(result_path, result);
        printf("Result saved: '%s'\n", result_path.c_str());
    }

    return 0;
}
