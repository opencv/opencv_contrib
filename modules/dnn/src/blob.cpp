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

            int size[3] = { cn, rows, cols };
            this->create(3, size, dstType);
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

    inline void squeezeShape_(const int srcDims, const int *srcSizes, const int dstDims, int *dstSizes)
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

    static Vec4i squeezeShape4(const int ndims, const int *sizes)
    {
        Vec4i res;
        squeezeShape_(ndims, sizes, 4, &res[0]);
        return res;
    }

    void Blob::fill(int ndims, const int *sizes, int type, void *data, bool deepCopy)
    {
        CV_Assert(type == CV_32F || type == CV_64F);

        Vec4i shape = squeezeShape4(ndims, sizes);

        if (deepCopy)
        {
            m.create(4, &shape[0], type);
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

        //TODO
        *this = Blob(in);
    }

    void Blob::create(int ndims, const int *sizes, int type)
    {
        CV_Assert(type == CV_32F || type == CV_64F);
        Vec4i shape = squeezeShape4(ndims, sizes);
        m.create(shape.channels, &shape[0], type);
    }

    void Blob::create(Vec4i shape, int type)
    {
        m.create(shape.channels, &shape[0], type);
    }

    void Blob::create(int num, int cn, int rows, int cols, int type)
    {
        Vec4i shape(num, cn, rows, cols);
        create(4, &shape[0], type);
    }

    Vec4i Blob::shape4() const
    {
        return squeezeShape4(dims(), sizes());
    }


}
}