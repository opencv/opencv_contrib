#include <opencv2/core.hpp>
#include <opencv2/stereo.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    std::string parameterFileLocation = "./parameters.yaml";
    if (argc > 1)
        parameterFileLocation = argv[1];


    Ptr<stereo::QuasiDenseStereo> stereo =  stereo::QuasiDenseStereo::create(cv::Size(5,5));
    stereo->saveParameters(parameterFileLocation);

    return 0;
}
