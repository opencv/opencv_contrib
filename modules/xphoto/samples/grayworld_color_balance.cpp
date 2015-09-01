#include "opencv2/xphoto.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "opencv2/core/utility.hpp"

using namespace cv;
using namespace std;

const char* keys =
{
    "{i || input image name}"
    "{o || output image name}"
};

int main( int argc, const char** argv )
{
    bool printHelp = ( argc == 1 );
    printHelp = printHelp || ( argc == 2 && string(argv[1]) == "--help" );
    printHelp = printHelp || ( argc == 2 && string(argv[1]) == "-h" );

    if ( printHelp )
    {
        printf("\nThis sample demonstrates the grayworld balance algorithm\n"
            "Call:\n"
            "    simple_color_blance -i=in_image_name [-o=out_image_name]\n\n");
        return 0;
    }

    CommandLineParser parser(argc, argv, keys);
    if ( !parser.check() )
    {
        parser.printErrors();
        return -1;
    }

    string inFilename = parser.get<string>("i");
    string outFilename = parser.get<string>("o");

    Mat src = imread(inFilename, 1);
    if ( src.empty() )
    {
        printf("Cannot read image file: %s\n", inFilename.c_str());
        return -1;
    }

    Mat res(src.size(), src.type());
    xphoto::autowbGrayworld(src, res);

    if ( outFilename == "" )
    {
        namedWindow("after white balance", 1);
        imshow("after white balance", res);

        waitKey(0);
    }
    else
        imwrite(outFilename, res);

    return 0;
}
