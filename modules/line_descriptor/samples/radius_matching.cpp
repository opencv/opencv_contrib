#include <opencv2/line_descriptor.hpp>

#include "opencv2/core/utility.hpp"
#include "opencv2/core/private.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>

using namespace cv;

static const std::string images[] =
{
    "cameraman.jpg",
    "church.jpg",
    "church2.png",
    "einstein.jpg",
    "stuff.jpg"
};

static const char* keys =
{
    "{@image_path | | Image path }"
};

static void help()
{
  std::cout << "\nThis example shows the functionalities of radius matching " <<
          "Please, run this sample using a command in the form\n" <<
          "./example_line_descriptor_radius_matching <path_to_input_images>/"
          << std::endl;
}

int main( int argc, char** argv )
{
    /* get parameters from comand line */
    CommandLineParser parser( argc, argv, keys );
    String pathToImages = parser.get<String>( 0 );

    /* create structures for hosting KeyLines and descriptors */
    int num_elements = sizeof( images ) / sizeof( images[0] );
    std::vector<Mat> descriptorsMat;
    std::vector<std::vector<KeyLine> > linesMat;

    /*create a pointer to a BinaryDescriptor object */
    Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();

    /* compute lines and descriptors */
    for(int i = 0; i<num_elements; i++)
    {
        /* get path to image */
        std::stringstream image_path;
        image_path << pathToImages << images[i];

        /* load image */
        Mat loadedImage = imread(image_path.str().c_str(), 1);
        if(loadedImage.data == NULL)
        {
            std::cout << "Could not load images." << std::endl;
            help();
            exit(-1);
        }

        /* compute lines and descriptors */
        std::vector<KeyLine> lines;
        Mat computedDescr;
        bd->detect(loadedImage, lines);
        bd->compute(loadedImage, lines, computedDescr);

        descriptorsMat.push_back(computedDescr);
        linesMat.push_back(lines);

    }

    /* compose a queries matrix */
    Mat queries;
    for(size_t j = 0; j<descriptorsMat.size(); j++)
    {
        if(descriptorsMat[j].rows >= 5)
            queries.push_back(descriptorsMat[j].rowRange(0, 5));

        else if(descriptorsMat[j].rows >0 && descriptorsMat[j].rows<5)
            queries.push_back(descriptorsMat[j]);
    }

    std::cout << "It has been generated a matrix of " << queries.rows
              << " descriptors" << std::endl;

    /* create a BinaryDescriptorMatcher object */
    Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

    /* populate matcher */
    bdm->add(descriptorsMat);

    /* compute matches */
    std::vector<std::vector<DMatch> > matches;
    bdm->radiusMatch(queries, matches, 30);

    /* print matches */
    for(size_t q = 0; q<matches.size(); q++)
    {
        for(size_t m = 0; m<matches[q].size(); m++)
        {
            DMatch dm = matches[q][m];
            std::cout << "Descriptor: " << q << " Image: " << dm.imgIdx
                      << " Distance: " << dm.distance << std::endl;
        }
    }
}
