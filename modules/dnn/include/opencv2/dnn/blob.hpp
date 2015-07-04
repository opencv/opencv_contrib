#ifndef __OPENCV_DNN_DNN_BLOB_HPP__
#define __OPENCV_DNN_DNN_BLOB_HPP__
#include <opencv2/core.hpp>
#include <vector>

namespace cv
{
namespace dnn
{
    /** @brief provides convenient methods for continuous n-dimensional array processing, dedicated for convolution neural networks
    It's realized as wrapper over \ref cv::Mat and \ref cv::UMat and will support methods for CPU/GPU switching
    */
    class CV_EXPORTS Blob
    {
    public:
        explicit Blob();
        explicit Blob(InputArray in);

        void create(int ndims, const int *sizes, int type = CV_32F);
        void create(Vec4i shape, int type = CV_32F);
        void create(int num, int cn, int rows, int cols, int type = CV_32F);

        void fill(InputArray in);
        void fill(int ndims, const int *sizes, int type, void *data, bool deepCopy = true);

        Mat& getMatRef();
        const Mat& getMatRef() const;
        Mat getMat();
        Mat getMat(int n, int cn);

        //shape getters
        ///returns real count of blob dimensions
        int dims() const;

        /** @brief returns size of corresponding dimension (axis)
        @param axis dimension index
        Python-like indexing is supported, so \p axis can be negative, i. e. -1 is last dimension.
        Supposed that size of non-existing dimensions equal to 1, so the method always finished.
        */
        int size(int axis) const;

        /** @brief returns size of corresponding dimension (axis)
        @param axis dimension index
        Python-like indexing is supported, so \p axis can be negative, i. e. -1 is last dimension.
        @note Unlike ::size, if \p axis points to non-existing dimension then an error will be generated.
        */
        int sizeAt(int axis) const;
        
        /** @brief returns number of elements
        @param startAxis starting axis (inverse indexing can be used)
        @param endAxis ending (excluded) axis
        @see ::canonicalAxis
        */
        size_t total(int startAxis = 0, int endAxis = -1) const;

        /** @brief converts axis index to canonical format (where 0 <= axis <= ::dims)
        */
        int canonicalAxis(int axis) const;

        /** @brief returns real shape of the blob
        */
        std::vector<int> shape() const;

        //shape getters, oriented for 4-dim Blobs processing
        int cols() const;
        int rows() const;
        int channels() const;
        int num() const;
        Size size2() const;
        Vec4i shape4() const;

        //CPU data pointer functions
        int offset(int n = 0, int cn = 0, int row = 0, int col = 0) const;
        uchar *ptrRaw(int n = 0, int cn = 0, int row = 0, int col = 0);
        float *ptrf(int n = 0, int cn = 0, int row = 0, int col = 0);
        template<typename TFloat>
        TFloat *ptr(int n = 0, int cn = 0, int row = 0, int col = 0);

        int type() const;
        bool isFloat() const;
        bool isDouble() const;

    private:
        const int *sizes() const;

        Mat m;
    };

    //////////////////////////////////////////////////////////////////////////

    inline int Blob::canonicalAxis(int axis) const
    {
        CV_Assert(-dims() <= axis && axis < dims());

        if (axis < 0)
        {
            return dims() + axis;
        }
        return axis;
    }

    inline int Blob::size(int axis) const
    {
        if (axis < 0)
            axis += dims();
        
        if (axis < 0 || axis >= dims())
            return 1;
        
        return sizes()[axis];
    }

    inline int Blob::sizeAt(int axis) const
    {
        CV_Assert(-dims() <= axis && axis < dims());

        if (axis < 0)
            axis += dims();

        return sizes()[axis];
    }

    inline size_t Blob::total(int startAxis, int endAxis) const
    {
        startAxis = canonicalAxis(startAxis);

        if (endAxis == -1)
            endAxis = dims();

        CV_Assert(startAxis <= endAxis && endAxis <= dims());

        size_t size = 1; //assume that blob isn't empty
        for (int i = startAxis; i < endAxis; i++)
            size *= (size_t)sizes()[i];

        return size;
    }

    inline int Blob::offset(int n, int cn, int row, int col) const
    {
        CV_DbgAssert(0 <= n && n < num() && 0 <= cn && cn < channels() && 0 <= row && row < rows() && 0 <= col && col < cols());
        return ((n*channels() + cn)*rows() + row)*cols() + col;
    }

    inline float *Blob::ptrf(int n, int cn, int row, int col)
    {
        CV_Assert(type() == CV_32F);
        return (float*)m.data + offset(n, cn, row, col);
    }

    inline uchar *Blob::ptrRaw(int n, int cn, int row, int col)
    {
        return m.data + m.elemSize() * offset(n, cn, row, col);
    }

    template<typename TFloat>
    inline TFloat* Blob::ptr(int n, int cn, int row, int col)
    {
        CV_Assert(type() == cv::DataDepth<TFloat>::value);
        return (TFloat*) ptrRaw(n, cn, row, col);
    }

    inline std::vector<int> Blob::shape() const
    {
        return std::vector<int>(sizes(), sizes() + dims());
    }

    inline Mat& Blob::getMatRef()
    {
        return m;
    }

    inline const Mat& Blob::getMatRef() const
    {
        return m;
    }

    inline Mat Blob::getMat()
    {
        return m;
    }

    inline Mat Blob::getMat(int n, int cn)
    {
        return Mat(rows(), cols(), m.type(), this->ptrRaw(n, cn));
    }

    inline int Blob::cols() const
    {
        return size(-1);
    }

    inline int Blob::rows() const
    {
        return size(-2);
    }

    inline Size Blob::size2() const
    {
        return Size(cols(), rows());
    }

    inline int Blob::channels() const
    {
        return size(-3);
    }

    inline int Blob::num() const
    {
        return size(-4);
    }

    inline int Blob::type() const
    {
        return m.depth();
    }

    inline bool Blob::isFloat() const
    {
        return (type() == CV_32F);
    }

    inline bool Blob::isDouble() const
    {
        return (type() == CV_32F);
    }

    inline const int * Blob::sizes() const
    {
        return &m.size[0];
    }

    inline int Blob::dims() const
    {
        return m.dims;
    }
}
}

#endif
