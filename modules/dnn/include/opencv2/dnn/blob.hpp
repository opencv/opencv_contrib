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
//! @addtogroup dnn
//! @{

    /** @brief Lightweight class for storing and processing a shape of blob (or anything else). */
    struct BlobShape
    {
        BlobShape();                                        //!< Creates [1, 1, 1, 1] shape @todo Make more clearer behavior.
        BlobShape(int s0);                                  //!< Creates 1-dim shape [@p s0]
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

        /** @brief Returns the product of all sizes of axes. */
        ptrdiff_t total();

        /** @brief Returns pointer to the first element of continuous size array. */
        const int *ptr() const;

        /** @brief Checks equality of two shapes. */
        bool equal(const BlobShape &other) const;

        bool operator== (const BlobShape &r) const;

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
        explicit Blob();

        /** @brief Constructs blob with specified @p shape and @p type. */
        explicit Blob(const BlobShape &shape, int type = CV_32F);

        /** @brief Constucts 4-dimensional blob (so-called batch) from image or array of images.
         * @param image 2-dimensional multi-channel or 3-dimensional single-channel image (or array of images)
         * @param dstCn specify size of second axis of ouptut blob
        */
        explicit Blob(InputArray image, int dstCn = -1);

        /** @brief Creates blob with specified @p shape and @p type. */
        void create(const BlobShape &shape, int type = CV_32F);

        /** @brief Creates blob from cv::Mat or cv::UMat without copying the data */
        void fill(InputArray in);
        /** @brief Creates blob from user data.
         *  @details If @p deepCopy is false then CPU data will not be allocated.
         */
        void fill(const BlobShape &shape, int type, void *data, bool deepCopy = true);

        Mat& matRef();                      //!< Returns reference to cv::Mat, containing blob data.
        const Mat& matRefConst() const;     //!< Returns reference to cv::Mat, containing blob data, for read-only purposes.
        UMat &umatRef();                    //!< Returns reference to cv::UMat, containing blob data (not implemented yet).
        const UMat &umatRefConst() const;   //!< Returns reference to cv::UMat, containing blob data, for read-only purposes (not implemented yet).

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

        /** @brief Converts @p axis index to canonical format (where 0 <= axis < dims()). */
        int canonicalAxis(int axis) const;

        /** @brief Returns shape of the blob. */
        BlobShape shape() const;

        /** @brief Checks equality of two blobs shapes. */
        bool equalShape(const Blob &other) const;

        /** @brief Returns slice of first two dimensions.
         *  @details The behaviour is similar to the following numpy code: blob[n, cn, ...]
         */
        Mat getPlane(int n, int cn);

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
        template<typename TFloat>
        TFloat *ptr(int n = 0, int cn = 0, int row = 0, int col = 0);
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

        /** @brief Returns type of the blob. */
        int type() const;

    private:
        const int *sizes() const;

        Mat m;
    };

//! @}
}
}

#include "blob.inl.hpp"

#endif
