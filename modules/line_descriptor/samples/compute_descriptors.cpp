#include <opencv2/line_descriptor.hpp>


#include "opencv2/core/utility.hpp"
#include "opencv2/core/private.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

static const char* keys =
{
    "{@image_path | | Image path }"
};

static void help()
{
  cout << "\nThis example shows the functionalities of lines extraction " <<
          "and descriptors computation furnished by BinaryDescriptor class\n" <<
          "Please, run this sample using a command in the form\n" <<
          "./example_line_descriptor_compute_descriptors <path_to_input_image>" << endl;
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
    String image_path = parser.get<String>( 0 );

    if(image_path.empty())
    {
        help();
        return -1;
    }

    /* load image */
    cv::Mat imageMat = imread(image_path, 0);
    if(imageMat.data == NULL)
    {
        std::cout << "Error, image could not be loaded. Please, check its path" << std::endl;
    }

    /* create a random binary mask */
//    cv::Mat mask(imageMat.size(), CV_8UC1);
//    cv::randu(mask, Scalar::all(0), Scalar::all(1));
    cv::Mat mask = Mat::ones(imageMat.size(), CV_8UC1);

    /* create a pointer to a BinaryDescriptor object with default parameters */
    Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();

    /* compute lines */
    std::vector<KeyLine> keylines;
    bd->detect(imageMat, keylines, mask);

    std::vector<KeyLine> octave0;
    for(size_t i = 0; i<keylines.size(); i++)
    {
        if(keylines[i].octave == 0)
            octave0.push_back(keylines[i]);
    }

    /* compute descriptors */
    cv::Mat descriptors;

    bd->compute(imageMat, octave0, descriptors);
    writeMat(descriptors, "old_code", 0);

}
