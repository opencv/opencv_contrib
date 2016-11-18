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

#include "precomp.hpp"

#include "grayscale_bitmap.hpp"

namespace cv
{
    namespace xfeatures2d
    {
        namespace pct_signatures
        {
            GrayscaleBitmap::GrayscaleBitmap(InputArray _bitmap, int bitsPerPixel)
                : mBitsPerPixel(bitsPerPixel)
            {
                Mat bitmap = _bitmap.getMat();
                if (bitmap.empty())
                {
                    CV_Error(Error::StsBadArg, "Input bitmap is empty");
                }
                if (bitmap.depth() != CV_8U && bitmap.depth() != CV_16U)
                {
                    CV_Error(Error::StsUnsupportedFormat, "Input bitmap depth must be CV_8U or CV_16U");
                }
                if (bitmap.depth() == CV_8U)
                {
                    bitmap.convertTo(bitmap, CV_16U, 257);
                }

                Mat grayscaleBitmap;
                cvtColor(bitmap, grayscaleBitmap, COLOR_BGR2GRAY);

                mWidth = bitmap.cols;
                mHeight = bitmap.rows;

                if (bitsPerPixel <= 0 || bitsPerPixel > 8)
                {
                    CV_Error_(Error::StsBadArg, ("Invalid number of bits per pixel %d. Only values in range [1..8] are accepted.", bitsPerPixel));
                }

                // Allocate space for pixel data.
                int pixelsPerItem = 32 / mBitsPerPixel;
                mData.resize((mWidth*mHeight + pixelsPerItem - 1) / pixelsPerItem);

                // Convert the bitmap to grayscale and fill the pixel data.
                CV_Assert(grayscaleBitmap.depth() == CV_16U);
                for (int y = 0; y < mHeight; y++)
                {
                    for (int x = 0; x < mWidth; x++)
                    {
                        uint grayVal = ((uint)grayscaleBitmap.at<ushort>((int)y, (int)x)) >> (16 - mBitsPerPixel);
                        setPixel(x, y, grayVal);
                    }
                }
                // Prepare the preallocated contrast matrix for contrast-entropy computations
                mCoOccurrenceMatrix.resize((int)(1 << (mBitsPerPixel * 2)));   // mCoOccurrenceMatrix size = maxPixelValue^2
            }


            // HOT PATH 30%
            void GrayscaleBitmap::getContrastEntropy(int x, int y, float& contrast, float& entropy, int radius)
            {
                int fromX = (x > radius) ? x - radius : 0;
                int fromY = (y > radius) ? y - radius : 0;
                int toX = std::min<int>(mWidth - 1, x + radius + 1);
                int toY = std::min<int>(mHeight - 1, y + radius + 1);
                for (int j = fromY; j < toY; ++j)
                {
                    for (int i = fromX; i < toX; ++i)                               // for each pixel in the window
                    {
                        updateCoOccurrenceMatrix(getPixel(i, j), getPixel(i, j + 1));        // match every pixel with all 8 its neighbours
                        updateCoOccurrenceMatrix(getPixel(i, j), getPixel(i + 1, j));
                        updateCoOccurrenceMatrix(getPixel(i, j), getPixel(i + 1, j + 1));
                        updateCoOccurrenceMatrix(getPixel(i + 1, j), getPixel(i, j + 1));    // 4 updates per pixel in the window
                    }
                }

                contrast = 0.0;
                entropy = 0.0;

                uint pixelsScale = 1 << mBitsPerPixel;
                float normalizer = (float)((toX - fromX) * (toY - fromY) * 4);            // four increments per pixel in the window (see above)
                for (int j = 0; j < (int)pixelsScale; ++j)                              // iterate row in a 2D histogram
                {
                    for (int i = 0; i <= j; ++i)                                        // iterate column up to the diagonal in 2D histogram
                    {
                        if (mCoOccurrenceMatrix[j*pixelsScale + i] != 0)                // consider only non-zero values
                        {
                            float value = (float)mCoOccurrenceMatrix[j*pixelsScale + i] / normalizer; // normalize value by number of histogram updates
                            contrast += (i - j) * (i - j) * value;          // compute contrast
                            entropy -= value * std::log(value);             // compute entropy
                            mCoOccurrenceMatrix[j*pixelsScale + i] = 0;     // clear the histogram array for the next computation
                        }
                    }
                }
            }


            void GrayscaleBitmap::convertToMat(OutputArray _bitmap, bool normalize) const
            {
                _bitmap.create((int)getHeight(), (int)getWidth(), CV_8U);
                Mat bitmap = _bitmap.getMat();

                for (int y = 0; y < getHeight(); ++y)
                {
                    for (int x = 0; x < getWidth(); ++x)
                    {
                        uint pixel = getPixel(x, y);
                        if (normalize && mBitsPerPixel < 8)
                        {
                            pixel <<= 8 - mBitsPerPixel;
                        }
                        else if (normalize && mBitsPerPixel > 8)
                        {
                            pixel >>= mBitsPerPixel - 8;
                        }
                        bitmap.at<uchar>((int)y, (int)x) = (uchar)pixel;
                    }
                }
            }

        }
    }
}
