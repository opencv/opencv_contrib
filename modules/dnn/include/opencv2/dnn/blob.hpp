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
    /** @brief Lightweight class for storing and processing a shape of blob (or anything else).
     */
    struct BlobShape
    {
        explicit BlobShape(int ndims = 4, int fill = 1);    //!< Creates n-dim shape and fill its by @p fill
        BlobShape(int num, int cn, int rows, int cols);     //!< Creates 4-dim shape [@p num, @p cn, @p rows, @p cols]
        BlobShape(int ndims, const int *sizes);             //!< Creates n-dim shape from the @p sizes array
        BlobShape(const std::vector<int> &sizes);           //!< Creates n-dim shape from the @p sizes vector
        template<int n>
        BlobShape(const Vec<int, n> &shape);                //!< Creates n-dim shape from @ref cv::Vec

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
         * @see size()
         */
        int size(int axis) const;

        int operator[](int axis) const; //!< Does the same thing as size(axis).
        int &operator[](int axis);      //!< Does the same thing as size(int) const.

        /** @brief Returns the size of the specified @p axis.
         *
         * Does the same thing as size(int) const, but if non-existing axis will be passed then 1 will be returned,
         * therefore this function always finishes successfuly.
         */
        int xsize(int axis) const;

        /** @brief Returns the product of all sizes of axes. */
        ptrdiff_t total();

        /** @brief Returns pointer to the first element of continuous size array. */
        const int *ptr() const;

        /** @brief Checks equality of two shapes. */
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

        /** @brief Constructs blob with specified @p shape and @p type. */
        explicit Blob(const BlobShape &shape, int type = CV_32F);

        /** @brief Constucts 4-dimensional blob from image or array of images.
         * @param image 2-dimensional multi-channel or 3-dimensional single-channel image (or array of images)
         * @param dstCn if specified force size of ouptut blob channel-dimension
        */
        explicit Blob(InputArray image, int dstCn = -1);

        /** @brief Creates blob with specified @p shape and @p type. */
        void create(const BlobShape &shape, int type = CV_32F);

        void fill(InputArray in);
        void fill(const BlobShape &shape, int type, void *data, bool deepCopy = true);

        Mat& getMatRef();
        const Mat& getMatRef() const;
        //TODO: add UMat get methods

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
         * therefore this function always finishes successfuly.
         */
        int xsize(int axis) const;

        /** @brief Computes the product of sizes of axes among the specified axes range [@p startAxis; @p endAxis).
         * @param startAxis the first axis to include in the range.
         * @param endAxis   the first axis to exclude from the range.
         * @details Negative axis indexing can be used.
         * @see canonicalAxis()
         */
        size_t total(int startAxis = 0, int endAxis = INT_MAX) const;

        /** @brief converts @p axis index to canonical format (where 0 <= axis < dims())
        */
        int canonicalAxis(int axis) const;

        /** @brief Returns shape of the blob. */
        BlobShape shape() const;

        /** @brief Checks equality of two blobs shapes. */
        bool equalShape(const Blob &other) const;

        /** @brief Returns sclice of first two dimensions.
         *  @details The behavior is similar to the following numpy code: blob[n, cn, ...]
         */
        Mat getPlane(int n, int cn);

        /** @addtogroup Shape getters of 4-dimensional blobs.
         *  @{
         */
        int cols() const;       //!< Returns size of the fourth blob axis.
        int rows() const;       //!< Returns size of the thrid  blob axis.
        int channels() const;   //!< Returns size of the second blob axis.
        int num() const;        //!< Returns size of the first blob axis.
        Size size2() const;     //!< Returns cv::Size(cols(), rows())
        Vec4i shape4() const;   //!< Returns shape of firt four blob axes.
        /** @}*/

        /** @addtogroup CPU pointer getters
         *  @{
         */
        template<int n>
        size_t offset(const Vec<int, n> &pos) const;
        size_t offset(int n = 0, int cn = 0, int row = 0, int col = 0) const;
        uchar *ptr(int n = 0, int cn = 0, int row = 0, int col = 0);
        template<typename TFloat>
        TFloat *ptr(int n = 0, int cn = 0, int row = 0, int col = 0);
        /** Returns (float*) ptr() */
        float *ptrf(int n = 0, int cn = 0, int row = 0, int col = 0);
        //TODO: add const ptr methods
        /** @}*/

        /** @brief Shares data from other @p blob.
         * @returns *this
         */
        Blob &shareFrom(const Blob &blob);

        /** @brief Changes shape of the blob without copying the data.
         * @returns *this
         */
        Blob &reshape(const BlobShape &shape);

        /** @brief Returns type of the blob. */
        int type() const;

    private:
        const int *sizes() const;

        Mat m;
    };
}
}

#include "blob.inl.hpp"

#endif
