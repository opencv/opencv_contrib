#include "opencv2/dnn.hpp"

namespace cv
{
namespace dnn
{    

    Blob::Blob(Mat &in) : _InputOutputArray(in)
    {
        
    }

    Blob::Blob(const Mat &in) : _InputOutputArray(in)
    {

    }

    Blob::Blob(UMat &in) : _InputOutputArray(in)
    {
        
    }

    Blob::Blob(const UMat &in) : _InputOutputArray(in)
    {

    }
}
}