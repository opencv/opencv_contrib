#include "precomp.hpp"

namespace cv
{
namespace dnn
{

    Blob::Blob()
    {
        int zeros[4] = { 0, 0, 0, 0 };
        m = Mat(4, zeros, CV_32F, NULL);
    }

    Blob::Blob(InputArray in)
    {
        CV_Assert(in.isMat() || in.isUMat());

        if (in.isMat())
        {
            Mat mat = in.getMat();

            CV_Assert(mat.dims == 2);
            int rows = mat.rows;
            int cols = mat.cols;
            int cn = mat.channels();
            int type = mat.type();
            int dstType = CV_MAKE_TYPE(CV_MAT_DEPTH(type), 1);

            this->create(BlobShape(1, cn, rows, cols), dstType);
            uchar *data = m.data;
            int step = rows * cols * CV_ELEM_SIZE(dstType);

            if (cn == 1)
            {
                Mat wrapper2D(rows, cols, dstType, m.data);
                mat.copyTo(wrapper2D);
            }
            else
            {
                std::vector<Mat> wrappers(cn);
                for (int i = 0; i < cn; i++)
                {
                    wrappers[i] = Mat(rows, cols, dstType, data);
                    data += step;
                }

                cv::split(mat, wrappers);
            }
        }
        else
        {
            CV_Error(cv::Error::StsNotImplemented, "Not Implemented");
        }
    }

    void Blob::fill(const BlobShape &shape, int type, void *data, bool deepCopy)
    {
        CV_Assert(type == CV_32F || type == CV_64F);

        if (deepCopy)
        {
            m.create(shape.dims(), shape.ptr(), type);
            memcpy(m.data, data, m.total() * m.elemSize());
        }
        else
        {
            m = Mat(shape.dims(), shape.ptr(), type, data);
        }
    }

    void Blob::fill(InputArray in)
    {
        CV_Assert(in.isMat() || in.isMatVector());

        //TODO
        *this = Blob(in);
    }

    void Blob::create(const BlobShape &shape, int type)
    {
        CV_Assert(type == CV_32F || type == CV_64F);
        m.create(shape.dims(), shape.ptr(), type);
    }

    inline void squeezeShape(const int srcDims, const int *srcSizes, const int dstDims, int *dstSizes)
    {
        const int m = std::min(dstDims, srcDims);

        //copy common(last) dimensions
        for (int i = 0; i < m; i++)
            dstSizes[dstDims - 1 - i] = srcSizes[srcDims - 1 - i];

        //either flatten extra dimensions
        for (int i = m; i < srcDims; i++)
            dstSizes[0] *= srcSizes[srcDims - 1 - i];

        //either fill gaps
        for (int i = m; i < dstDims; i++)
            dstSizes[dstDims - 1 - i] = 1;
    }

    Vec4i Blob::shape4() const
    {
        return Vec4i(num(), channels(), rows(), cols());
    }

    std::ostream &operator<< (std::ostream &stream, const BlobShape &shape)
    {
        stream << "[";

        for (int i = 0; i < shape.dims() - 1; i++)
            stream << shape[i] << ", ";
        if (shape.dims() > 0)
            stream << shape[-1];

        return stream << "]";
    }
}
}
