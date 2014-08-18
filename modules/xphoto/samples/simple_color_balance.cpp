#include "opencv2/xphoto.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/types_c.h"

const char* keys =
{
    "{i || input image name}"
    "{o || output image name}"
};

int main( int argc, const char** argv )
{
    bool printHelp = ( argc == 1 );
    printHelp = printHelp || ( argc == 2 && std::string(argv[1]) == "--help" );
    printHelp = printHelp || ( argc == 2 && std::string(argv[1]) == "-h" );

    if ( printHelp )
    {
        printf("\nThis sample demonstrates simple color balance algorithm\n"
            "Call:\n"
            "    simple_color_blance -i=in_image_name [-o=out_image_name]\n\n");
        return 0;
    }

    cv::CommandLineParser parser(argc, argv, keys);
    if ( !parser.check() )
    {
        parser.printErrors();
        return -1;
    }

    std::string inFilename = parser.get<std::string>("i");
    std::string outFilename = parser.get<std::string>("o");

    cv::Mat src = cv::imread(inFilename, 1);
    if ( src.empty() )
    {
        printf("Cannot read image file: %s\n", inFilename.c_str());
        return -1;
    }

    cv::Mat res(src.size(), src.type());
    cv::xphoto::balanceWhite(src, res, cv::xphoto::WHITE_BALANCE_SIMPLE);

    if ( outFilename == "" )
    {
        cv::namedWindow("after white balance", 1);
        cv::imshow("after white balance", res);

        cv::waitKey(0);
    }
    else
        cv::imwrite(outFilename, res);

    return 0;
}