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

namespace cv
{
namespace dnn
{
    struct BlobShape
    {
        explicit BlobShape(int ndims = 4, int fill = 1);
        BlobShape(int num, int cn, int rows, int cols);
        BlobShape(int ndims, const int *sizes);
        BlobShape(const std::vector<int> &sizes);
        template<int n>
        BlobShape(const Vec<int, n> &shape);

        int dims() const;
        int size(int axis) const;
        int &size(int axis);

        //do the same as size()
        int operator[](int axis) const;
        int &operator[](int axis);

        //same as size(), but size of non-existing dimensions equal to 1
        int xsize(int axis) const;

        ptrdiff_t total();

        const int *ptr() const;

        bool equal(const BlobShape &other) const;

    private:
        cv::AutoBuffer<int,4> sz;
    };

    bool operator== (const BlobShape &l, const BlobShape &r);

    //maybe useless
    CV_EXPORTS std::ostream &operator<< (std::ostream &stream, const BlobShape &shape);


    /** @brief Provides convenient methods for continuous n-dimensional array processing, dedicated for convolution neural networks.
     *
     * It's realized as wrapper over @ref cv::Mat and @ref cv::UMat and will support methods for CPU/GPU switching.
    */
    class CV_EXPORTS Blob
    {
    public:
        explicit Blob();

        explicit Blob(const BlobShape &shape, int type = CV_32F);

        /** @brief constucts 4-dimensional blob from input
         *  @param in 2-dimensional or 3-dimensional single-channel image (or vector from them)
         *  @param dstCn if specified force size of ouptut blob channel-dimension
        */
        explicit Blob(InputArray in, int dstCn = -1);

        void create(const BlobShape &shape, int type = CV_32F);

        void fill(InputArray in);
        void fill(const BlobShape &shape, int type, void *data, bool deepCopy = true);

        Mat& getMatRef();
        const Mat& getMatRef() const;
        //TODO: add UMat get methods
        Mat getMat(int n, int cn);

        //shape getters
        ///returns real count of blob dimensions
        int dims() const;

        /** @brief returns size of corresponding dimension (axis)
        @param axis dimension index
        Python-like indexing is supported, so @p axis can be negative, i. e. -1 is last dimension.
        Supposed that size of non-existing dimensions equal to 1, so the method always finished.
        */
        int xsize(int axis) const;

        /** @brief returns size of corresponding dimension (axis)
        @param axis dimension index
        Python-like indexing is supported, so @p axis can be negative, i. e. -1 is last dimension.
        @note Unlike xsize(), if @p axis points to non-existing dimension then an error will be generated.
        */
        int size(int axis) const;

        /** @brief returns number of elements
        @param startAxis starting axis (inverse indexing can be used)
        @param endAxis ending (excluded) axis
        @see canonicalAxis()
        */
        size_t total(int startAxis = 0, int endAxis = -1) const;

        /** @brief converts axis index to canonical format (where 0 <= axis < dims())
        */
        int canonicalAxis(int axis) const;

        /** @brief returns shape of the blob
        */
        BlobShape shape() const;

        bool equalShape(const Blob &other) const;

        //shape getters for 4-dim Blobs processing
        int cols() const;
        int rows() const;
        int channels() const;
        int num() const;
        Size size2() const;
        Vec4i shape4() const;

        //CPU data pointer functions
        template<int n>
        size_t offset(const Vec<int, n> &pos) const;
        size_t offset(int n = 0, int cn = 0, int row = 0, int col = 0) const;
        uchar *ptr(int n = 0, int cn = 0, int row = 0, int col = 0);
        template<typename TFloat>
        TFloat *ptr(int n = 0, int cn = 0, int row = 0, int col = 0);
        float *ptrf(int n = 0, int cn = 0, int row = 0, int col = 0);
        //TODO: add const ptr methods

        /** @brief Share data with other blob.
        @returns *this
        */
        Blob &shareFrom(const Blob &blob);

        /** @brief Adjust blob shape to required (data reallocated if needed).
        @returns *this
        */
        Blob &reshape(const BlobShape &shape);

        int type() const;
        bool isFloat() const;
        bool isDouble() const;

    private:
        const int *sizes() const;

        Mat m;
    };
}
}

#include "blob.inl.hpp"

#endif
