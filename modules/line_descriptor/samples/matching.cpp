/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2014, Biagio Montesano, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include <opencv2/line_descriptor.hpp>

#include "opencv2/core/utility.hpp"
#include "opencv2/core/private.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;

static const char* keys =
{
    "{@image_path1 | | Image path 1 }"
    "{@image_path2 | | Image path 2 }"
};

static void help()
{
  std::cout << "\nThis example shows the functionalities of lines extraction " <<
          "and descriptors computation furnished by BinaryDescriptor class\n" <<
          "Please, run this sample using a command in the form\n" <<
          "./example_line_descriptor_compute_descriptors <path_to_input_image 1>"
          << "<path_to_input_image 2>" << std::endl;

}

int main( int argc, char** argv )
{
    /* get parameters from comand line */
    CommandLineParser parser( argc, argv, keys );
    String image_path1 = parser.get<String>( 0 );
    String image_path2 = parser.get<String>( 1 );

    if(image_path1.empty() || image_path2.empty())
    {
        help();
        return -1;
    }


    /* load image */
    cv::Mat imageMat1 = imread(image_path1, 1);
    cv::Mat imageMat2 = imread(image_path2, 1);

    waitKey();
    if(imageMat1.data == NULL || imageMat2.data == NULL)
    {
        std::cout << "Error, images could not be loaded. Please, check their path"
                  << std::endl;
    }

    /* create binary masks */
    cv::Mat mask1 = Mat::ones(imageMat1.size(), CV_8UC1);
    cv::Mat mask2 = Mat::ones(imageMat2.size(), CV_8UC1);

    /* create a pointer to a BinaryDescriptor object with default parameters */
    Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();

    /* compute lines */
    std::vector<KeyLine> keylines1, keylines2;
    bd->detect(imageMat1, keylines1, mask1);
    bd->detect(imageMat2, keylines2, mask2);

    /* compute descriptors */
    cv::Mat descr1, descr2;
    bd->compute(imageMat1, keylines1, descr1);
    bd->compute(imageMat2, keylines2, descr2);

    /* create a BinaryDescriptorMatcher object */
    Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

    /* require match */
    std::vector<DMatch> matches;
    bdm->match(descr1, descr2, matches);

    /* plot matches */
    cv::Mat outImg;
    std::vector<char> mask (matches.size(), 1);
    drawLineMatches(imageMat1, keylines1, imageMat2, keylines2, matches,
                outImg, Scalar::all(-1), Scalar::all(-1), mask,
                DrawLinesMatchesFlags::DEFAULT);

    std::cout << "num dmatch " << matches.size() << std::endl;
    imshow("Matches", outImg);
    waitKey();
}


