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
// Copyright (C) 2015, University of Ostrava, Institute for Research and Applications of Fuzzy Modeling,
// Pavel Vlasanek, all rights reserved. Third party copyrights are property of their respective owners.
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

#ifndef __OPENCV_IMG_HASH_H__
#define __OPENCV_IMG_HASH_H__

#include "opencv2/img_hash/img_hash_base.hpp"
#include "opencv2/img_hash/average_hash.hpp"
#include "opencv2/img_hash/block_mean_hash.hpp"
#include "opencv2/img_hash/color_moment_hash.hpp"
#include "opencv2/img_hash/marr_hildreth_hash.hpp"
#include "opencv2/img_hash/phash.hpp"
#include "opencv2/img_hash/radial_variance_hash.hpp"

/**
@defgroup img_hash Provide algorithms to extract the hash of images and fast way to figure out most similar images in huge data set

Namespace for all functions is **img_hash**. The module brings implementations of different image hashing.

  @{
    @defgroup avg_hash Simple and fast perceptual hash algorithm

    This is a fast image hashing algorithm, but only work on simple case.For more details, please
    refer to http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

    @defgroup p_hash Slower than average_hash, but tolerant of minor modifications

    This algorithm can combat more variation than averageHash, for more details please refer to
    http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

    @defgroup marr_hash Marr-Hildreth Operator Based Hash, slowest but more discriminative
    http://www.phash.org/docs/pubs/thesis_zauner.pdf

    @defgroup radial_var_hash Image hash based on Radon transform.
    http://www.phash.org/docs/pubs/thesis_zauner.pdf

    @defgroup block_mean_hash Image hash based on block mean.
    http://www.phash.org/docs/pubs/thesis_zauner.pdf

    @defgroup color_moment_hash Image hash based on color moments.
    http://www.naturalspublishing.com/files/published/54515x71g3omq1.pdf
   @}

*/

#endif // __OPENCV_IMG_HASH_H__
