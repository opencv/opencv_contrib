#include "opencv2/dnn.hpp"
using namespace cv;
using namespace cv::dnn;

namespace cv
{
namespace dnn
{

Blob::Blob()
{

}

Blob::Blob(InputArray in)
{
    CV_Assert(in.isMat());
    m = in.getMat();
}

Net::~Net()
{

}


Importer::~Importer()
{

}


Ptr<NetConfiguration> NetConfiguration::create()
{
    return Ptr<NetConfiguration>(new NetConfiguration());
}

}
}