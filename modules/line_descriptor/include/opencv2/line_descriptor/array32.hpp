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

#ifndef __OPENCV_ARRAY32_HPP
#define __OPENCV_ARRAY32_HPP

#include "types.hpp"

class Array32
{

 private:
  static double ARRAY_RESIZE_FACTOR;
  static double ARRAY_RESIZE_ADD_FACTOR;

 public:
  /* set ARRAY_RESIZE_FACTOR */
  static void setArrayResizeFactor( double arf );

  /* constructor */
  Array32();

  /* destructor */
  ~Array32();

  /* cleaning function used in destructor */
  void cleanup();

  /* push data */
  void push( UINT32 data );

  /* insert data at given index */
  void insert( UINT32 index, UINT32 data );

  /* return data */
  UINT32* data();

  /* return data size */
  UINT32 size();

  /* return capacity */
  UINT32 capacity();

  /* definition of operator = */
  void operator=( const Array32& );

  /* print data */
  void print();

  /* initializer */
  void init( int size );

  /* data */
  UINT32 *arr;

};

#endif
