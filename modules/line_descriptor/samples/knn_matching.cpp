#include <opencv2/line_descriptor.hpp>

#include "opencv2/core/utility.hpp"
#include "opencv2/core/private.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>

using namespace cv;

static const char* keys =
{
    "{@image_path1 | | Image path 1 }"
    "{@image_path2 | | Image path 2 }"
};

static void help()
{
  std::cout << "\nThis example shows the functionalities of descriptors matching\n" <<
          "Please, run this sample using a command in the form\n" <<
          "./example_line_descriptor_matching <path_to_input_image 1>"
          << "<path_to_input_image 2>" << std::endl;

}

/* invert numBits bits in input char */
uchar invertSingleBits (uchar dividend_char, int numBits)
{
    std::vector<int> bin_vector;
    long dividend;
    long bin_num;

    /* convert input char to a long */
    dividend = (long)dividend_char;

    /*if a 0 has been obtained, just generate a 8-bit long vector of zeros */
    if(dividend == 0)
        bin_vector = std::vector<int>(8, 0);

    /* else, apply classic decimal to binary conversion */
    else
    {
        while ( dividend >= 1 )
        {
            bin_num = dividend % 2;
            dividend /= 2;
            bin_vector.push_back(bin_num);
        }
    }

    /* ensure that binary vector always has length 8 */
    if(bin_vector.size()<8){
        std::vector<int> zeros (8-bin_vector.size(), 0);
        bin_vector.insert(bin_vector.end(), zeros.begin(), zeros.end());
    }

    /* invert numBits bits */
    for(int index = 0; index<numBits; index++)
    {
        if(bin_vector[index] == 0)
            bin_vector[index] = 1;

        else
            bin_vector[index] = 0;
    }

    /* reconvert to decimal */
    uchar result;
    for(int i = (int)bin_vector.size()-1; i>=0; i--)
        result += bin_vector[i]*pow(2, i);

    return result;
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

    if(imageMat1.data == NULL || imageMat2.data == NULL)
    {
        std::cout << "Error, images could not be loaded. Please, check their paths"
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

    /* make a copy of descr2 mat */
    Mat descr2Copy = descr1.clone();

    /* randomly change some bits in original descriptors */
    srand (time(NULL));

    for(int j = 0; j<descr1.rows; j++)
    {
        /* select a random column */
        int randCol = rand() % 32;

        /* get correspondent data */
        uchar u = descr1.at<uchar>(j, randCol);

        /* change bits */
        for(int k = 1; k<=5; k++)
        {
            /* copy current row to train matrix */
            descr2Copy.push_back(descr1.row(j));

            /* invert k bits */
            uchar uc = invertSingleBits(u, k);

            /* update current row in train matrix */
            descr2Copy.at<uchar>(descr2Copy.rows-1, randCol) = uc;
        }
    }

    /* prepare a structure to host matches */
    std::vector<std::vector<DMatch> > matches;

    /* require knn match */
    bdm->knnMatch(descr1, descr2, matches, 6);

    /* visualize matches and Hamming distances */
    for(size_t v = 0; v<matches.size(); v++)
    {
        for(size_t m = 0; m<matches[v].size(); m++)
        {
            DMatch dm = matches[v][m];
            std::cout << dm.queryIdx << " " << dm.trainIdx << " "
                      << dm.distance << std::endl;
        }
    }

}


