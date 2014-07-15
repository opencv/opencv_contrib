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

/* constructor */
BucketGroup::BucketGroup()
{
  empty = 0;
  group = NULL;
}

/* destructor */
BucketGroup::~BucketGroup()
{
  if( group != NULL )
    delete group;
}

/* insert data into the bucket */
void BucketGroup::insert( int subindex, UINT32 data )
{
  if( group == NULL )
  {
    group = new Array32();
    group->push( 0 );
  }

  UINT32 lowerbits = ( (UINT32) 1 << subindex ) - 1;
  int end = popcnt( empty & lowerbits );

  if( ! ( empty & ( (UINT32) 1 << subindex ) ) )
  {
    group->insert( end, group->arr[end + 2] );
    empty |= (UINT32) 1 << subindex;
  }

  int totones = popcnt( empty );
  group->insert( totones + 1 + group->arr[2 + end + 1], data );
  for ( int i = end + 1; i < totones + 1; i++ )
    group->arr[2 + i]++;
}

/* perform a query to the bucket */
UINT32* BucketGroup::query( int subindex, int *size )
{
  if( empty & ( (UINT32) 1 << subindex ) )
  {
    UINT32 lowerbits = ( (UINT32) 1 << subindex ) - 1;
    int end = popcnt( empty & lowerbits );
    int totones = popcnt( empty );
    *size = group->arr[2 + end + 1] - group->arr[2 + end];
    return group->arr + 2 + totones + 1 + group->arr[2 + end];
  }

  else
  {
    *size = 0;
    return NULL;
  }
}
