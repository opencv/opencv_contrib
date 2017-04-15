/**************************************************************************************
The structered edge demo requires you to provide a model.
This model can be found at the opencv_extra repository on Github on the following link:
https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz
***************************************************************************************/

#include <opencv2/ximgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ximgproc;

const char* keys =
{
    "{i || input image name}"
    "{m || model name}"
    "{o || output image name}"
};

int main( int argc, const char** argv )
{
    bool printHelp = ( argc == 1 );
    printHelp = printHelp || ( argc == 2 && String(argv[1]) == "--help" );
    printHelp = printHelp || ( argc == 2 && String(argv[1]) == "-h" );

    if ( printHelp )
    {
        std::cout << "\nThis sample demonstrates structured forests for fast edge detection\n"
               "Call:\n"
               "    structured_edge_detection -i=in_image_name -m=model_name [-o=out_image_name]\n\n";
        return 0;
    }

    CommandLineParser parser(argc, argv, keys);
    if ( !parser.check() )
    {
        parser.printErrors();
        return -1;
    }

    String modelFilename = parser.get<String>("m");
    String inFilename = parser.get<String>("i");
    String outFilename = parser.get<String>("o");

    Mat image = imread(inFilename, 1);
    if ( image.empty() )
        CV_Error(Error::StsError, String("Cannot read image file: ") + inFilename);

    if ( modelFilename.size() == 0)
        CV_Error(Error::StsError, String("Empty model name"));

    image.convertTo(image, DataType<float>::type, 1/255.0);

    Mat edges(image.size(), image.type());

    Ptr<StructuredEdgeDetection> pDollar =
        createStructuredEdgeDetection(modelFilename);
    pDollar->detectEdges(image, edges);

    if ( outFilename.size() == 0 )
    {
        namedWindow("edges", 1);
        imshow("edges", edges);
        waitKey(0);
    }
    else
        imwrite(outFilename, 255*edges);

    return 0;
}
