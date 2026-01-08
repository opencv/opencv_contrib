#include "opencv2/xphoto.hpp"
#include "opencv2/highgui.hpp"

const char* keys =
{
    "{i || input image name}"
    "{o || output image name}"
    "{sigma || expected noise standard deviation}"
    "{tw |4| template window size}"
    "{sw |16| search window size}"
};

int main(int argc, const char** argv)
{
    bool printHelp = (argc == 1);
    printHelp = printHelp || (argc == 2 && std::string(argv[1]) == "--help");
    printHelp = printHelp || (argc == 2 && std::string(argv[1]) == "-h");

    if (printHelp)
    {
        printf("\nThis sample demonstrates BM3D image denoising\n"
            "Call:\n"
            "    bm3d_image_denoising -i=<string> -sigma=<double> -tw=<int> -sw=<int> [-o=<string>]\n\n");
        return 0;
    }

    cv::CommandLineParser parser(argc, argv, keys);
    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    std::string inFilename = parser.get<std::string>("i");
    std::string outFilename = parser.get<std::string>("o");

    cv::Mat src = cv::imread(inFilename, cv::IMREAD_GRAYSCALE);
    if (src.empty())
    {
        printf("Cannot read image file: %s\n", inFilename.c_str());
        return -1;
    }

    float sigma = parser.get<float>("sigma");
    if (sigma == 0.0)
        sigma = 15.0;

    int templateWindowSize = parser.get<int>("tw");
    if (templateWindowSize == 0)
        templateWindowSize = 4;

    int searchWindowSize = parser.get<int>("sw");
    if (searchWindowSize == 0)
        searchWindowSize = 16;

    cv::Mat res(src.size(), src.type());
    cv::xphoto::bm3dDenoising(src, res, sigma, templateWindowSize, searchWindowSize);

    if (outFilename.empty())
    {
        cv::namedWindow("input image", cv::WINDOW_NORMAL);
        cv::imshow("input image", src);
        cv::namedWindow("denoising result", cv::WINDOW_NORMAL);
        cv::imshow("denoising result", res);
        cv::waitKey(0);
    }
    else
    {
        cv::imwrite(outFilename, res);
    }

    return 0;
}
