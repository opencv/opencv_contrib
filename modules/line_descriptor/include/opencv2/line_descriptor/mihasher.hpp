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
 // Copyright (C) 2014, Mohammad Norouzi, Ali Punjani, David J. Fleet,
 // all rights reserved.
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

#ifndef __OPENCV_MIHASHER_HPP
#define __OPENCV_MIHASHER_HPP

#ifdef _WIN32
#pragma warning( disable : 4267 )
#endif

#include "types.hpp"
#include "bitops.hpp"
#include "sparse_hashtable.hpp"
#include "bitarray.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class Mihasher
{
 private:

  /* Bits per code */
  int B;

  /* B/8 */
  int B_over_8;

  /* Bits per chunk (must be less than 64) */
  int b;

  /* Number of chunks */
  int m;

  /* Number of chunks with b bits (have 1 bit more than others) */
  int mplus;

  /* Maximum hamming search radius (we use B/2 by default) */
  int D;

  /* Maximum hamming search radius per substring */
  int d;

  /* Maximum results to return */
  int K;

  /* Number of codes */
  UINT64 N;

  /* Table of original full-length codes */
  cv::Mat codes;

  /* Counter for eliminating duplicate results (it is not thread safe) */
  bitarray *counter;

  /* Array of m hashtables */
  SparseHashtable *H;

  /* Volume of a b-bit Hamming ball with radius s (for s = 0 to d) */
  UINT32 *xornum;

  /* Used within generation of binary codes at a certain Hamming distance */
  int power[100];

 public:

  /* constructor */
  Mihasher();

  /* desctructor */
  ~Mihasher();

  /* constructor 2 */
  Mihasher( int B, int m );

  /* K setter */
  void setK( int K );

  /* populate tables */
  void populate( cv::Mat & codes, UINT32 N, int dim1codes );

  /* execute a batch query */
  void batchquery( UINT32 * results, UINT32 *numres/*, qstat *stats*/, const cv::Mat & q, UINT32 numq, int dim1queries );

 private:

  /* execute a single query */
  void query( UINT32 * results, UINT32* numres/*, qstat *stats*/, UINT8 *q, UINT64 * chunks, UINT32 * res );
};

#endif
