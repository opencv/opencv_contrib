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

#include "precomp.hpp"

/* no need for the static keyword in the definition */
double Array32::ARRAY_RESIZE_FACTOR = 1.1;    // minimum is 1.0
double Array32::ARRAY_RESIZE_ADD_FACTOR = 4;  // minimum is 1

/* set ARRAY_RESIZE_FACTOR */
void Array32::setArrayResizeFactor( double arf )
{
  ARRAY_RESIZE_FACTOR = arf;
}

/* constructor */
Array32::Array32()
{
  arr = NULL;
}

/* definition of operator =
 Array32& Array32::operator = (const Array32 &rhs) */
void Array32::operator =( const Array32 &rhs )
{
  if( &rhs != this )
    this->arr = rhs.arr;
}

/* destructor */
Array32::~Array32()
{
  cleanup();
}

/* cleaning function used in destructor */
void Array32::cleanup()
{
  free( arr );
}

/* push data */
void Array32::push( UINT32 Data )
{
  if( arr )
  {
    if( arr[0] == arr[1] )
    {
      arr[1] = std::max( ceil( arr[1] * ARRAY_RESIZE_FACTOR ), arr[1] + ARRAY_RESIZE_ADD_FACTOR );
      UINT32* new_Data = static_cast<UINT32*>( realloc( arr, sizeof(UINT32) * ( 2 + arr[1] ) ) );
      if( new_Data == NULL )
      {
        /* could not realloc, but orig still valid */
        std::cout << "ALERT!!!! Not enough memory, operation aborted!" << std::endl;
        exit( 0 );
      }
      else
      {
        arr = new_Data;
      }

    }

    arr[2 + arr[0]] = Data;
    arr[0]++;

  }

  else
  {
    arr = (UINT32*) malloc( ( 2 + ARRAY_RESIZE_ADD_FACTOR ) * sizeof(UINT32) );
    arr[0] = 1;
    arr[1] = 1;
    arr[2] = Data;
  }
}

/* insert data at given index */
void Array32::insert( UINT32 index, UINT32 Data )
{
  if( arr )
  {
    if( arr[0] == arr[1] )
    {
      arr[1] = ceil( arr[0] * 1.1 );
      UINT32* new_data = static_cast<UINT32*>( realloc( arr, sizeof(UINT32) * ( 2 + arr[1] ) ) );
      if( new_data == NULL )
      {
        // could not realloc, but orig still valid
        std::cout << "ALERT!!!! Not enough memory, operation aborted!" << std::endl;
        exit( 0 );
      }
      else
      {
        arr = new_data;
      }
    }

    memmove( arr + ( 2 + index ) + 1, arr + ( 2 + index ), ( arr[0] - index ) * sizeof ( *arr ) );

    arr[2 + index] = Data;
    arr[0]++;
  }

  else
  {
    arr = (UINT32*) malloc( 3 * sizeof(UINT32) );
    arr[0] = 1;
    arr[1] = 1;
    arr[2] = Data;
  }
}

/* return data */
UINT32* Array32::data()
{
  return arr ? arr + 2 : NULL;
}

/* return data size */
UINT32 Array32::size()
{
  return arr ? arr[0] : 0;
}

/* return capacity */
UINT32 Array32::capacity()
{
  return arr ? arr[1] : 0;
}

/* print data */
void Array32::print()
{
  for ( int i = 0; i < (int) size(); i++ )
    printf( "%d, ", arr[i + 2] );

  printf( "\n" );
}

/* initializer */
void Array32::init( int Size )
{
  if( arr == NULL )
  {
    arr = (UINT32*) malloc( ( 2 + Size ) * sizeof(UINT32) );
    arr[0] = 0;
    arr[1] = Size;
  }
}
