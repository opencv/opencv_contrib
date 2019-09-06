// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/infoflow.hpp>

using namespace std;
using namespace cv;
using namespace cv::alphamat;

int check_image(Mat& image)
{
    if ( !image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl;
        return -1;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    Mat image, tmap;
    char* img_path = argv[1];
    image = imread(img_path, CV_LOAD_IMAGE_COLOR);   // Read the file
    check_image(image);

    char* tmap_path = argv[2];
    tmap = imread(tmap_path, CV_LOAD_IMAGE_GRAYSCALE);
    check_image(tmap);

    Mat result;
    infoFlow(image, tmap, result, false, true);
    imshow("Result Matte", result);
    return 0;
    
}
