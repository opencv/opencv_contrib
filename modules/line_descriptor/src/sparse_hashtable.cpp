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
 // Copyright (C) 2014, Biagio Montesano, all rights reserved.
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

const int SparseHashtable::MAX_B = 37;

/* constructor */
SparseHashtable::SparseHashtable()
{
  table = NULL;
  size = 0;
  b = 0;
}

/* initializer */
int SparseHashtable::init( int _b )
{
  b = _b;

  if( b < 5 || b > MAX_B || b > (int) ( sizeof(UINT64) * 8 ) )
    return 1;

  size = UINT64_1 << ( b - 5 );  // size = 2 ^ b
  table = (BucketGroup*) calloc( size, sizeof(BucketGroup) );

  return 0;

}

/* destructor */
SparseHashtable::~SparseHashtable()
{
  free( table );
}

/* insert data */
void SparseHashtable::insert( UINT64 index, UINT32 data )
{
  table[index >> 5].insert( (int) ( index % 32 ), data );
}

/* query data */
UINT32* SparseHashtable::query( UINT64 index, int *Size )
{
  return table[index >> 5].query( (int) ( index % 32 ), Size );
}
