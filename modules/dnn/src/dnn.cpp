#include "opencv2/dnn.hpp"
using namespace cv;
using namespace cv::dnn;
#include <algorithm>

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

static Vec4i blobNormalizeShape(int ndims, const int *sizes)
{
    Vec4i shape = Vec4i::all(1);

    for (int i = 0; i < std::min(3, ndims); i++)
        shape[3 - i] = sizes[ndims-1 - i];

    for (int i = 3; i < ndims; i++)
        shape[0] *= sizes[ndims-1 - i];

    return shape;
}

void Blob::fill(int ndims, const int *sizes, int type, void *data, bool deepCopy)
{
    CV_Assert(type == CV_32F || type == CV_64F);

    Vec4i shape = blobNormalizeShape(ndims, sizes);

    if (deepCopy)
    {
        m.create(3, &shape[0], type);
        size_t dataSize = m.total() * m.elemSize();
        memcpy(m.data, data, dataSize);
    }
    else
    {
        m = Mat(shape.channels, &shape[0], type, data);
    }
}

void Blob::fill(InputArray in)
{
    CV_Assert(in.isMat() || in.isMatVector());
}

void Blob::create(int ndims, const int *sizes, int type /*= CV_32F*/)
{
    Vec4i shape = blobNormalizeShape(ndims, sizes);
    m.create(shape.channels, &shape[0], type);
}

struct Net::Impl
{

};

Net::Net() : impl(new Net::Impl)
{

}

Net::~Net()
{

}

Importer::~Importer()
{

}

#include <sstream>
template<typename T>
String toString(const T &v)
{
    std::stringstream ss;
    ss << v;
    return ss.str();
}

int Layer::getNumInputs()
{
    return 1;
}

int Layer::getNumOutputs()
{
    return 1;
}

cv::String Layer::getInputName(int inputNum)
{
    return "input" + toString(inputNum);
}


cv::String Layer::getOutputName(int outputNum)
{
    return "output" + toString(outputNum);
}

Layer::~Layer()
{

}

}
}