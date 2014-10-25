#include "opencv2/xphoto.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/types_c.h"

const char* keys =
{
    "{i || input image name}"
    "{o || output image name}"
    "{sigma || expected noise standard deviation}"
    "{psize |16| expected noise standard deviation}"
};

int main( int argc, const char** argv )
{
    bool printHelp = ( argc == 1 );
    printHelp = printHelp || ( argc == 2 && std::string(argv[1]) == "--help" );
    printHelp = printHelp || ( argc == 2 && std::string(argv[1]) == "-h" );

    if ( printHelp )
    {
        printf("\nThis sample demonstrates dct-based image denoising\n"
            "Call:\n"
            "    dct_image_denoising -i=<string> -sigma=<double> -psize=<int> [-o=<string>]\n\n");
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

	double sigma = parser.get<double>("sigma");
	if (sigma == 0.0)
		sigma = 15.0;

    int psize = parser.get<int>("psize");
    if (psize == 0)
        psize = 16;

    cv::Mat res(src.size(), src.type());
    cv::xphoto::dctDenoising(src, res, sigma, psize);

    if ( outFilename == "" )
    {
        cv::namedWindow("denoising result", 1);
        cv::imshow("denoising result", res);

        cv::waitKey(0);
    }
    else
        cv::imwrite(outFilename, res);

    return 0;
}