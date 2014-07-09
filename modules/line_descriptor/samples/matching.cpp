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


