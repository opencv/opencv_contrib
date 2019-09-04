// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>

#include <opencv2/dnn_superres.hpp>
#include <opencv2/dnn_superres_quality.hpp>

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
        cout << "usage:   Arg 1: image path | Path to image" << endl;
        cout << "\t Arg 2: algorithm 1 | edsr, espcn, fsrcnn or lapsrn" << endl;
        cout << "\t Arg 3: path to model file 1 \n";
        cout << "\t Arg 4: scale     | 2, 3 ,4 or 8 \n";
        return -1;
    }

    string path = string(argv[1]);
    string algorithm = string(argv[2]);
    string model = string(argv[3]);
    int scale = atoi(argv[4]);

    Mat img = cv::imread(path);
    if (img.empty())
    {
        std::cerr << "Couldn't load image: " << img << "\n";
        return -2;
    }

    std::cout << "Image loaded." << std::endl;

    //Crop the image so the images will be aligned
    int width = img.cols - (img.cols % scale);
    int height = img.rows - (img.rows % scale);
    Mat cropped = img(Rect(0, 0, width, height));

    //Downscale the image for benchmarking
    Mat img_downscaled;
    cv::resize(cropped, img_downscaled, cv::Size(), 1.0/scale, 1.0/scale);

    //Make dnn super resolution instance
    DnnSuperResImpl sr;

    Mat img_new;

    //Read and set the dnn model
    sr.readModel(model);
    sr.setModel(algorithm, scale);
    sr.upsample(img_downscaled, img_new);

    //Perform and display benchmarking
    std::vector<double> psnrs, ssims, perfs;

    DnnSuperResQuality::setFontColor(cv::Scalar(255,0,0));
    DnnSuperResQuality::setFontScale(1.0);
    DnnSuperResQuality::setFontFace(cv::FONT_HERSHEY_COMPLEX_SMALL);
    DnnSuperResQuality::benchmark(sr, cropped, psnrs, ssims, perfs, 1, 1);

    return 0;
}