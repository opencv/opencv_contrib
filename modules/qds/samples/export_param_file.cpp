#include <opencv2/core.hpp>
#include <opencv2/qds.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    std::string parameterFileLocation = "";
    if (argc > 1)
    parameterFileLocation = argv[1];

    qds::QuasiDenseStereo(cv::Size(5,5)).saveParameters(parameterFileLocation);

    return 0;
}
