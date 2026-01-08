#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>

#include <iostream>

static inline cv::Mat operator& ( const cv::Mat& lhs, const cv::Matx23d& rhs )
{
    cv::Mat ret;
    cv::warpAffine ( lhs, ret, rhs, lhs.size(), cv::INTER_LINEAR );
    return ret;
}

static inline cv::Mat operator& ( const cv::Matx23d& lhs, const cv::Mat& rhs )
{
    cv::Mat ret;
    cv::warpAffine ( rhs, ret, lhs, rhs.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP );
    return ret;
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, "{ @input1 | ../data/peilin_plane.png | }{ @input2 | ../data/peilin_shape.png | }");
    parser.about("\nThis program demonstrates Pei&Lin Normalization\n");
    parser.printMessage();

    std::string filename1 = parser.get<std::string>("@input1");
    std::string filename2 = parser.get<std::string>("@input2");

    cv::Mat I = cv::imread(filename1, 0);
    if (I.empty())
    {
        std::cout << "Couldn't open image " << filename1 << std::endl;
        return 0;
    }
    cv::Mat J = cv::imread(filename2, 0);
    if (J.empty())
    {
        std::cout << "Couldn't open image " << filename2 << std::endl;
        return 0;
    }
    cv::Mat N = I & cv::ximgproc::PeiLinNormalization ( I );
    cv::Mat D = cv::ximgproc::PeiLinNormalization ( J ) & I;
    cv::imshow ( "I", I );
    cv::imshow ( "N", N );
    cv::imshow ( "J", J );
    cv::imshow ( "D", D );
    cv::waitKey();
    return 0;
}
