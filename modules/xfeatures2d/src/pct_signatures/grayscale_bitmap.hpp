/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2016, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

/*
Contributed by Gregor Kovalcik <gregor dot kovalcik at gmail dot com>
    based on code provided by Martin Krulis, Jakub Lokoc and Tomas Skopal.

References:
    Martin Krulis, Jakub Lokoc, Tomas Skopal.
    Efficient Extraction of Clustering-Based Feature Signatures Using GPU Architectures.
    Multimedia tools and applications, 75(13), pp.: 8071–8103, Springer, ISSN: 1380-7501, 2016

    Christian Beecks, Merih Seran Uysal, Thomas Seidl.
    Signature quadratic form distance.
    In Proceedings of the ACM International Conference on Image and Video Retrieval, pages 438-445.
    ACM, 2010.
*/


#ifndef _OPENCV_XFEATURES_2D_PCT_SIGNATURES_GRAYSCALE_BITMAP_HPP_
#define _OPENCV_XFEATURES_2D_PCT_SIGNATURES_GRAYSCALE_BITMAP_HPP_

#ifdef __cplusplus


namespace cv
{
    namespace xfeatures2d
    {
        namespace pct_signatures
        {
            /**
            * @brief Specific implementation of grayscale bitmap. This bitmap is used
            *       to compute contrast and entropy features from surroundings of a pixel.
            */
            class GrayscaleBitmap
            {
            public:
                /**
                * @brief Initialize the grayscale bitmap from regular bitmap.
                * @param bitmap Bitmap used as source of data.
                * @param bitsPerPixel How many bits occupy one pixel in grayscale (e.g., 8 ~ 256 grayscale values).
                *       Must be within [1..8] range.
                */
                GrayscaleBitmap(InputArray bitmap, int bitsPerPixel = 4);

                /**
                * @brief Return the width of the image in pixels.
                */
                int getWidth() const
                {
                    return mWidth;
                }

                /**
                * @brief Return the height of the image in pixels.
                */
                int getHeight() const
                {
                    return mHeight;
                }

                /**
                * @brief Return the height of the image in pixels.
                */
                int getBitsPerPixel() const
                {
                    return mBitsPerPixel;
                }


                /**
                * @brief Compute contrast and entropy at selected coordinates.
                * @param x The horizontal coordinate of the pixel (0..width-1).
                * @param y The vertical coordinate of the pixel (0..height-1).
                * @param contrast Output variable where contrast value is saved.
                * @param entropy Output variable where entropy value is saved.
                * @param radius Radius of the rectangular window around selected pixel used for computing
                *       contrast and entropy. Size of the window side is (2*radius + 1).
                *       The window is cropped if [x,y] is too near the image border.
                */
                void getContrastEntropy(
                    int x,
                    int y,
                    float& contrast,
                    float& entropy,
                    int windowRadius = 3);

                /**
                * @brief Converts to OpenCV CV_8U Mat for debug and visualization purposes.
                * @param bitmap OutputArray proxy where Mat will be written.
                * @param normalize Whether to normalize the bitmap to the whole range of 255 values
                *       to improve contrast.
                */
                void convertToMat(OutputArray bitmap, bool normalize = true) const;


            private:
                /**
                * @brief Width of the image.
                */
                int mWidth;

                /**
                * @brief Height of the image.
                */
                int mHeight;

                /**
                * @brief Number of bits per pixel.
                */
                int mBitsPerPixel;

                /**
                * @brief Pixel data packed in 32-bit uints.
                */
                std::vector<uint> mData;

                /**
                * @brief Tmp matrix used for computing contrast and entropy.
                */
                std::vector<uint> mCoOccurrenceMatrix;


                /**
                * @brief Get pixel from packed data vector.
                * @param x The horizontal coordinate of the pixel (0..width-1).
                * @param y The vertical coordinate of the pixel (0..height-1).
                * @return Grayscale value (0 ~ black, 2^bitPerPixel - 1 ~ white).
                */
                uint inline getPixel(int x, int y) const
                {
                    int pixelsPerItem = 32 / mBitsPerPixel;
                    int offset = y*mWidth + x;
                    int shift = (offset % pixelsPerItem) * mBitsPerPixel;
                    uint mask = (1 << mBitsPerPixel) - 1;
                    return (mData[offset / pixelsPerItem] >> shift) & mask;
                }

                /**
                * @brief Set pixel in the packed data vector.
                * @param x The horizontal coordinate of the pixel (0..width-1).
                * @param y The vertical coordinate of the pixel (0..height-1).
                * @param val Grayscale value (0 ~ black, 2^bitPerPixel - 1 ~ white).
                */
                void inline setPixel(int x, int y, uint val)
                {
                    int pixelsPerItem = 32 / mBitsPerPixel;
                    int offset = y*mWidth + x;
                    int shift = (offset % pixelsPerItem) * mBitsPerPixel;
                    uint mask = (1 << mBitsPerPixel) - 1;
                    val &= mask;
                    mData[offset / pixelsPerItem] &= ~(mask << shift);
                    mData[offset / pixelsPerItem] |= val << shift;
                }

                /**
                * @brief Perform an update of contrast matrix.
                */
                void inline updateCoOccurrenceMatrix(uint a, uint b)
                {
                    // co-occurrence matrix is symmetric
                    // merge to a variable with greater higher bits
                    // to accumulate just in upper triangle in co-occurrence matrix for efficiency
                    int offset = (int)((a > b) ? (a << mBitsPerPixel) + b : a + (b << mBitsPerPixel));
                    mCoOccurrenceMatrix[offset]++;
                }


            };
        }
    }
}


#endif

#endif
