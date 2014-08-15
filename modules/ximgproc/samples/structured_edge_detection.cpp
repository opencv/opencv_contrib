#include <opencv2/ximgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

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
    printHelp = printHelp || ( argc == 2 && std::string(argv[1]) == "--help" );
    printHelp = printHelp || ( argc == 2 && std::string(argv[1]) == "-h" );

    if ( printHelp )
    {
        printf("\nThis sample demonstrates structured forests for fast edge detection\n"
               "Call:\n"
               "    structured_edge_detection -i=in_image_name -m=model_name [-o=out_image_name]\n\n");
        return 0;
    }

    cv::CommandLineParser parser(argc, argv, keys);
    if ( !parser.check() )
    {
        parser.printErrors();
        return -1;
    }

    std::string modelFilename = parser.get<std::string>("m");
    std::string inFilename = parser.get<std::string>("i");
    std::string outFilename = parser.get<std::string>("o");

    cv::Mat image = cv::imread(inFilename, 1);
    if ( image.empty() )
    {
        printf("Cannot read image file: %s\n", inFilename.c_str());
        return -1;
    }

    image.convertTo(image, cv::DataType<float>::type, 1/255.0);

    cv::Mat edges(image.size(), image.type());

    cv::Ptr<StructuredEdgeDetection> pDollar =
        createStructuredEdgeDetection(modelFilename);
    pDollar->detectEdges(image, edges);

    if ( outFilename == "" )
    {
        cv::namedWindow("edges", 1);
        cv::imshow("edges", edges);

        cv::waitKey(0);
    }
    else
        cv::imwrite(outFilename, 255*edges);

    return 0;
}
