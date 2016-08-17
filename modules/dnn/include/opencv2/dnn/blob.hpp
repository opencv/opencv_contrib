/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_DNN_DNN_BLOB_HPP__
#define __OPENCV_DNN_DNN_BLOB_HPP__
#include <opencv2/core.hpp>
#include <vector>
#include <ostream>
#include <iostream>

namespace cv
{
namespace dnn
{
//! @addtogroup dnn
//! @{

    /** @brief Lightweight class for storing and processing a shape of blob (or anything else). */
    struct CV_EXPORTS_W BlobShape
    {
        BlobShape();                                        //!< Creates [1, 1, 1, 1] shape @todo Make more clearer behavior.
        explicit BlobShape(int s0);                         //!< Creates 1-dim shape [@p s0]
        BlobShape(int s0, int s1);                          //!< @overload
        BlobShape(int s0, int s1, int s2);                  //!< @overload
        BlobShape(int num, int cn, int rows, int cols);     //!< Creates 4-dim shape [@p num, @p cn, @p rows, @p cols]

        //! Creates n-dim shape from the @p sizes array; if @p sizes is NULL then shape will contain unspecified data
        BlobShape(int ndims, const int *sizes);
        BlobShape(const std::vector<int> &sizes);           //!< Creates n-dim shape from the @p sizes vector
        template<int n>
        BlobShape(const Vec<int, n> &shape);                //!< Creates n-dim shape from @ref cv::Vec

        //! Creates n-dim shape and fill its by @p fill
        static BlobShape all(int ndims, int fill = 1);

        /** @brief Returns number of dimensions. */
        int dims() const;

        /** @brief Returns reference to the size of the specified @p axis.
         *
         * Negative @p axis is supported, in this case a counting starts from the last axis,
         * i. e. -1 corresponds to last axis.
         * If non-existing axis was passed then an error will be generated.
         */
        int &size(int axis);

        /** @brief Returns the size of the specified @p axis.
         *  @see size()
         */
        int size(int axis) const;

        int operator[](int axis) const; //!< Does the same thing as size(axis).
        int &operator[](int axis);      //!< Does the same thing as size(int) const.

        /** @brief Returns the size of the specified @p axis.
         *
         * Does the same thing as size(int) const, but if non-existing axis will be passed then 1 will be returned,
         * therefore this function always finishes successfully.
         */
        int xsize(int axis) const;

        /** @brief Converts @p axis index to canonical format (where 0 <= @p axis < dims()). */
        int canonicalAxis(int axis) const;

        /** @brief Returns the product of all sizes of axes. */
        ptrdiff_t total() const;

        /** @brief Computes the product of sizes of axes among the specified axes range [@p startAxis; @p endAxis).
         * @details Negative axis indexing can be used. @sa Blob::total(int,int)
         */
        ptrdiff_t total(int startAxis, int endAxis = INT_MAX) const;

        /** @brief Constructs new shape from axes in range [@p startAxis; @p endAxis).
         * @details Negative axis indexing can be used. @sa Blob::total(int,int)
         */
        BlobShape slice(int startAxis, int endAxis = INT_MAX) const;

        /** @brief Returns pointer to the first element of continuous size array. */
        const int *ptr() const;
        /** @overload */
        int *ptr();

        bool equal(const BlobShape &other) const;       //!< Checks equality of two shapes.
        bool operator== (const BlobShape &r) const;     //!< @sa equal()

        BlobShape operator+ (const BlobShape &r) const; //!< Contacenates two shapes.

        static BlobShape like(const Mat &m);    //!< Returns shape of passed Mat.
        static BlobShape like(const UMat &m);   //!< Returns shape of passed UMat.

        static BlobShape empty();               //!< Returns empty shape [].
        bool isEmpty() const;                   //!< Returns true if shape is empty (i.e []).

#ifdef CV_CXX_MOVE_SEMANTICS
        //TBD
#endif

    private:
        cv::AutoBuffer<int,4> sz;
    };


    /** @brief This class provides methods for continuous n-dimensional CPU and GPU array processing.
     *
     * The class is realized as a wrapper over @ref cv::Mat and @ref cv::UMat.
     * It will support methods for switching and logical synchronization between CPU and GPU.
    */
    class CV_EXPORTS_W Blob
    {
    public:
        Blob();

        /** @brief Constructs blob with specified @p shape and @p type. */
        explicit Blob(const BlobShape &shape, int type = CV_32F, int allocFlags = ALLOC_MAT);

        /** @brief Constructs Blob from existing Mat or UMat. */
        Blob(InputArray data);

        /** @brief Constructs 4-dimensional blob (so-called batch) from image or array of images.
         * @param image 2-dimensional multi-channel or 3-dimensional single-channel image (or array of such images)
         * @param dstCn specifies size of second axis of ouptut blob
         */
        static Blob fromImages(InputArray image, int dstCn = -1);

        /** @brief Works like Blob::fromImages() but in-place. */
        void batchFromImages(InputArray image, int dstCn = -1);

        /** @brief Creates blob with specified @p shape and @p type. */
        void create(const BlobShape &shape, int type = CV_32F, int allocFlags = ALLOC_MAT);

        /** @brief Creates blob from Mat or UMat without copying the data.
          * @details If in is Mat then Mat data is populated, otherwise - UMat.
          */
        void fill(InputArray in);

        /** @brief Creates blob from user data.
         *  @details If @p deepCopy is false then CPU data will not be allocated.
         */
        void fill(const BlobShape &shape, int type, void *data, bool deepCopy = true);

        /** @brief Sets @p value to the last used data (if @p allocFlags = -1).
         * @details If @p allocFlags != -1 then destination data (Mat or UMat) is determined by flags from AllocFlag enum like in create().
         */
        void setTo(InputArray value, int allocFlags = -1);

        Mat& matRef(bool writeOnly = true);     //!< Returns reference to cv::Mat, containing blob data.
        const Mat& matRefConst() const;         //!< Returns reference to cv::Mat, containing blob data, for read-only purposes.
        UMat &umatRef(bool writeOnly = true);   //!< Returns reference to cv::UMat, containing blob data.
        const UMat &umatRefConst() const;       //!< Returns reference to cv::UMat, containing blob data, for read-only purposes.

        template<typename XMat>
        XMat &getRef(bool writeOnly = true);
        template<typename XMat>
        const XMat &getRefConst() const;

        void updateMat(bool syncData = true) const;     //!< Actualizes data stored inside Mat of Blob; if @p syncData is false then only shape will be actualized.
        void updateUMat(bool syncData = true) const;    //!< Actualizes data stored inside Mat of Blob; if @p syncData is false then only shape will be actualized.
        void sync() const;                              //!< Updates Mat and UMat of Blob.

        /** @brief Returns number of blob dimensions. */
        int dims() const;

        /** @brief Returns the size of the specified @p axis.
         *
         * Negative @p axis is supported, in this case a counting starts from the last axis,
         * i. e. -1 corresponds to last axis.
         * If non-existing axis was passed then an error will be generated.
         */
        int size(int axis) const;

        /** @brief Returns the size of the specified @p axis.
         *
         * Does the same thing as size(int) const, but if non-existing axis will be passed then 1 will be returned,
         * therefore this function always finishes successfully.
         */
        int xsize(int axis) const;

        /** @brief Computes the product of sizes of axes among the specified axes range [@p startAxis; @p endAxis).
         * @param startAxis the first axis to include in the range.
         * @param endAxis   the first axis to exclude from the range.
         * @details Negative axis indexing can be used.
         */
        size_t total(int startAxis = 0, int endAxis = INT_MAX) const;

        /** @brief Converts @p axis index to canonical format (where 0 <= @p axis < dims()). */
        int canonicalAxis(int axis) const;

        /** @brief Returns shape of the blob. */
        BlobShape shape() const;

        /** @brief Checks equality of two blobs shapes. */
        bool equalShape(const Blob &other) const;

        /** @brief Returns slice of first two dimensions.
         *  @details The behaviour is similar to the following numpy code: blob[n, cn, ...]
         */
        Mat getPlane(int n, int cn);

        /** @brief Returns slice of first dimension.
         *  @details The behaviour is similar to getPlane(), but returns all
         * channels * rows * cols values, corresponding to the n-th value
         * of the first dimension.
         */
        Mat getPlanes(int n);

        /* Shape getters of 4-dimensional blobs. */
        int cols() const;       //!< Returns size of the fourth axis blob.
        int rows() const;       //!< Returns size of the thrid  axis blob.
        int channels() const;   //!< Returns size of the second axis blob.
        int num() const;        //!< Returns size of the first  axis blob.
        Size size2() const;     //!< Returns cv::Size(cols(), rows())
        Vec4i shape4() const;   //!< Returns shape of first four blob axes.

        /** @brief Returns linear index of the element with specified coordinates in the blob.
         *
         * If @p n < dims() then unspecified coordinates will be filled by zeros.
         * If @p n > dims() then extra coordinates will be ignored.
         */
        template<int n>
        size_t offset(const Vec<int, n> &pos) const;
        /** @overload */
        size_t offset(int n = 0, int cn = 0, int row = 0, int col = 0) const;

        /* CPU pointer getters */
        /** @brief Returns pointer to the blob element with the specified position, stored in CPU memory.
         *
         * @p n correspond to the first axis, @p cn - to the second, etc.
         * If dims() > 4 then unspecified coordinates will be filled by zeros.
         * If dims() < 4 then extra coordinates will be ignored.
         */
        uchar *ptr(int n = 0, int cn = 0, int row = 0, int col = 0);
        /** @overload */
        template<typename Type>
        Type *ptr(int n = 0, int cn = 0, int row = 0, int col = 0);
        /** @overload ptr<float>() */
        float *ptrf(int n = 0, int cn = 0, int row = 0, int col = 0);
        //TODO: add const ptr methods

        /** @brief Shares data from other @p blob.
         * @returns *this
         */
        Blob &shareFrom(const Blob &blob);

        /** @brief Changes shape of the blob without copying the data.
         * @returns *this
         */
        Blob &reshape(const BlobShape &shape);

        /** @brief Changes shape of the blob without copying the data.
         * @returns shallow copy of original blob with new shape.
         */
        Blob reshaped(const BlobShape &newShape) const;

        int type() const;       //!< Returns type of the blob.
        int elemSize() const;   //!< Returns size of single element in bytes.
        int getState() const;   //!< Returns current state of the blob, @see DataState.

    private:
        const int *sizes() const;

#   define CV_DNN_UMAT //DBG
#ifdef HAVE_OPENCL
#   define CV_DNN_UMAT
#endif

#ifdef CV_DNN_UMAT
#   define CV_DNN_UMAT_ONLY(expr) (expr)
#else
#   define CV_DNN_UMAT_ONLY(expr)
#endif

#ifndef CV_DNN_UMAT
        Mat m;
#else
        mutable Mat m;
        mutable UMat um;
        mutable uchar state;
#endif

public:
        enum DataState
        {
            UNINITIALIZED   = 0,
            HEAD_AT_MAT     = 1 << 0,
            HEAD_AT_UMAT    = 1 << 1,
            SYNCED          = HEAD_AT_MAT | HEAD_AT_UMAT
        };

        enum AllocFlag
        {
            ALLOC_MAT  = HEAD_AT_MAT,
            ALLOC_UMAT = HEAD_AT_UMAT,
            ALLOC_BOTH = SYNCED
        };
    };

//! @}
}
}

#include "blob.inl.hpp"

#endif
