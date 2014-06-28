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

inline void writeMat(cv::Mat m, std::string name, int n)
{
    std::stringstream ss;
    std::string s;
    ss << n;
    ss >> s;
    std::string fileNameConf = name + s;
    cv::FileStorage fsConf(fileNameConf, cv::FileStorage::WRITE);
    fsConf << "m" << m;

    fsConf.release();
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
    cv::Mat imageMat1 = imread(image_path1, 0);
    cv::Mat imageMat2 = imread(image_path2, 0);
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

    std::cout << "lines " << keylines1.size() << " " << keylines2.size()
              << std::endl;

    /* compute descriptors */
    cv::Mat descr1, descr2;
    bd->compute(imageMat1, keylines1, descr1);
    bd->compute(imageMat2, keylines2, descr2);

    /* create a BinaryDescriptorMatcher object */
    Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

    /* require match */
    std::vector<DMatch> matches;
    bdm->match(descr1, descr2, matches);
    for(int x = 0; x<matches.size(); x++)
        std::cout << matches[x].queryIdx << " " << matches[x].trainIdx << std::endl;

    /* result checkout */
    cv::Mat result(descr1.size(), CV_8UC1);
    std::cout << "size " << descr1.rows << " " << descr1.cols
              << " " << descr2.rows << " " << descr2.cols << std::endl;
//    for(size_t i = 0; i<matches.size(); i++){
//        uchar* pointer = result.ptr(i);
//        uchar* trainPointer = descr2.ptr(matches[i].trainIdx);
//        *pointer = *trainPointer;
//        pointer++;
//    }


    /* write matrices */
    writeMat(descr1, "descr1", 0);
    writeMat(result, "result", 0);
}




































