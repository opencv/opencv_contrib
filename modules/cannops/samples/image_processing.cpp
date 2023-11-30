// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cann.hpp>
#include <opencv2/cann_interface.hpp>

int main(int argc, char* argv[])
{
    cv::CommandLineParser parser(argc, argv,
                                 "{@input|puppy.png|path to input image}"
                                 "{@output|output.png|path to output image}"
                                 "{help||show help}");
    parser.about("This is a sample for image processing with Ascend NPU. \n");
    if (argc != 3 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    std::string imagePath = parser.get<std::string>(0);
    std::string outputPath = parser.get<std::string>(1);

    // read input image and generate guass noise
    //! [input_noise]
    cv::Mat img = cv::imread(imagePath);
    // Generate gauss noise that will be added into the input image
    cv::Mat gaussNoise(img.rows, img.cols, img.type());
    cv::RNG rng;
    rng.fill(gaussNoise, cv::RNG::NORMAL, 0, 25);
    //! [input_noise]

    // setup cann
    //! [setup]
    cv::cann::initAcl();
    cv::cann::setDevice(0);
    //! [setup]

    //! [image-process]
    cv::Mat output;
    // add gauss noise to the image
    cv::cann::add(img, gaussNoise, output);
    // rotate the image with a certain mode (0, 1 and 2, correspond to rotation of 90, 180 and 270
    // degrees clockwise respectively)
    cv::cann::rotate(output, output, 0);
    // flip the image with a certain mode (0, positive and negative number, correspond to flipping
    // around the x-axis, y-axis and both axes respectively)
    cv::cann::flip(output, output, 0);
    //! [image-process]

    cv::imwrite(outputPath, output);

    //! [tear-down-cann]
    cv::cann::resetDevice();
    cv::cann::finalizeAcl();
    //! [tear-down-cann]
    return 0;
}
