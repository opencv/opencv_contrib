// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>

#include <opencv2/dnn_superres.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace dnn_superres;

int main(int argc, char *argv[])
{
    // Check for valid command line arguments, print usage
    // if insufficient arguments were given.
    if ( argc < 4 ) {
        cout << "usage:   Arg 1: image     | Path to image" << endl;
        cout << "\t Arg 2: algorithm | bilinear, bicubic, edsr, espcn, fsrcnn or lapsrn" << endl;
        cout << "\t Arg 3: scale     | 2, 3 or 4 \n";
        cout << "\t Arg 4: path to model file \n";
        return -1;
    }

    string img_path = string(argv[1]);
    string algorithm = string(argv[2]);
    int scale = atoi(argv[3]);
    string path = "";

    if( argc > 4)
        path = string(argv[4]);

    // Load the image
    Mat img = cv::imread(img_path);
    Mat original_img(img);
    if ( img.empty() )
    {
        std::cerr << "Couldn't load image: " << img << "\n";
        return -2;
    }

    //Make dnn super resolution instance
    DnnSuperResImpl sr;

    Mat img_new;

    if( algorithm == "bilinear" ){
        resize(img, img_new, Size(), scale, scale, 2);
    }
    else if( algorithm == "bicubic" )
    {
        resize(img, img_new, Size(), scale, scale, 3);
    }
    else if( algorithm == "edsr" || algorithm == "espcn" || algorithm == "fsrcnn" || algorithm == "lapsrn" )
    {
        sr.readModel(path);
        sr.setModel(algorithm, scale);
        sr.upsample(img, img_new);
    }
    else{
        std::cerr << "Algorithm not recognized. \n";
    }

    if ( img_new.empty() )
    {
        std::cerr << "Upsampling failed. \n";
        return -3;
    }
    cout << "Upsampling succeeded. \n";

    // Display image
    cv::namedWindow("Initial Image", WINDOW_AUTOSIZE);
    cv::imshow("Initial Image", img_new);
    //cv::imwrite("./saved.jpg", img_new);
    cv::waitKey(0);

    return 0;
}