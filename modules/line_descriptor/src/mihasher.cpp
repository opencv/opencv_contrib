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

/* execute a batch query */
void Mihasher::batchquery( UINT32 * results, UINT32 *numres, const cv::Mat & queries, UINT32 numq, int dim1queries )
{
  /* create and initialize a bitarray */
  counter = new bitarray;
  counter->init( N );

  UINT32 *res = new UINT32[K * ( D + 1 )];
  UINT64 *chunks = new UINT64[m];
  UINT32 * presults = results;
  UINT32 *pnumres = numres;

  /* make a copy of input queries */
  cv::Mat queries_clone = queries.clone();

  /* set a pointer to first query (row) */
  UINT8 *pq = queries_clone.ptr();

  /* loop over number of descriptors */
  for ( size_t i = 0; i < numq; i++ )
  {
    /* for every descriptor, query database */
    query( presults, pnumres, pq, chunks, res );

    /* move pointer to write next K indeces */
    presults += K;
    pnumres += B + 1;

    /* move forward pointer to current row in descriptors matrix */
    pq += dim1queries;

  }

  delete[] res;
  delete[] chunks;

  delete counter;
}

/* execute a single query */
void Mihasher::query( UINT32* results, UINT32* numres, UINT8 * Query, UINT64 *chunks, UINT32 *res )
{
  /* if K == 0 that means we want everything to be processed.
   So maxres = N in that case. Otherwise K limits the results processed */
  UINT32 maxres = K ? K : (UINT32) N;

  /* number of results so far obtained (up to a distance of s per chunk) */
  UINT32 n = 0;

  /* number of candidates tested with full codes (not counting duplicates) */
  UINT32 nc = 0;

  /* counting everything retrieved (duplicates are counted multiple times)
   number of lookups (and xors) */
  UINT32 nl = 0;

  UINT32 nd = 0;
  UINT32 *arr;
  int size = 0;
  UINT32 index;
  int hammd;

  counter->erase();
  memset( numres, 0, ( B + 1 ) * sizeof ( *numres ) );

  split( chunks, Query, m, mplus, b );

  /* the growing search radius per substring */
  int s;

  /* current b: for the first mplus substrings it is b, for the rest it is (b-1) */
  int curb = b;

  for ( s = 0; s <= d && n < maxres; s++ )
  {
    for ( int k = 0; k < m; k++ )
    {
      if( k < mplus )
        curb = b;
      else
        curb = b - 1;
      UINT64 chunksk = chunks[k];
      /* number of bit-strings with s number of 1s */
      nl += xornum[s + 1] - xornum[s];

      /* the bit-string with s number of 1s */
      UINT64 bitstr = 0;
      for ( int i = 0; i < s; i++ )
        /* power[i] stores the location of the i'th 1 */
        power[i] = i;
      /* used for stopping criterion (location of (s+1)th 1) */
      power[s] = curb + 1;

      /* bit determines the 1 that should be moving to the left */
      int bit = s - 1;

      /* start from the left-most 1, and move it to the left until
       it touches another one */

      /* the loop for changing bitstr */
      bool infiniteWhile = true;
      while ( infiniteWhile )
      {
        if( bit != -1 )
        {
          bitstr ^= ( power[bit] == bit ) ? (UINT64) 1 << power[bit] : (UINT64) 3 << ( power[bit] - 1 );
          power[bit]++;
          bit--;
        }

        else
        { /* bit == -1 */
          /* the binary code bitstr is available for processing */
          arr = H[k].query( chunksk ^ bitstr, &size );  // lookup
          if( size )
          { /* the corresponding bucket is not empty */
            nd += size;
            for ( int c = 0; c < size; c++ )
            {
              index = arr[c];
              if( !counter->get( index ) )
              { /* if it is not a duplicate */
                counter->set( index );
                hammd = match( codes.ptr() + (UINT64) index * ( B_over_8 ), Query, B_over_8 );

                nc++;
                if( hammd <= D && numres[hammd] < maxres )
                  res[hammd * K + numres[hammd]] = index + 1;

                numres[hammd]++;
              }
            }
          }

          /* end of processing */
          while ( ++bit < s && power[bit] == power[bit + 1] - 1 )
          {
            bitstr ^= (UINT64) 1 << ( power[bit] - 1 );
            power[bit] = bit;
          }
          if( bit == s )
            break;
        }
      }

      n = n + numres[s * m + k];
      if( n >= maxres )
        break;
    }
  }

  n = 0;
  for ( s = 0; s <= D && (int) n < K; s++ )
  {
    for ( int c = 0; c < (int) numres[s] && (int) n < K; c++ )
      results[n++] = res[s * K + c];
  }

}

/* constructor 2 */
Mihasher::Mihasher( int _B, int _m )
{
  B = _B;
  B_over_8 = B / 8;
  m = _m;
  b = (int) ceil( (double) B / m );

  /* assuming that B/2 is large enough radius to include
   all of the k nearest neighbors */
  D = (int) ceil( B / 2.0 );
  d = (int) ceil( (double) D / m );

  /* mplus is the number of chunks with b bits
   (m-mplus) is the number of chunks with (b-1) bits */
  mplus = B - m * ( b - 1 );

  xornum = new UINT32[d + 2];
  xornum[0] = 0;
  for ( int i = 0; i <= d; i++ )
    xornum[i + 1] = xornum[i] + (UINT32) choose( b, i );

  H = new SparseHashtable[m];

  /* H[i].init might fail */
  for ( int i = 0; i < mplus; i++ )
    H[i].init( b );
  for ( int i = mplus; i < m; i++ )
    H[i].init( b - 1 );
}

/* K setter */
void Mihasher::setK( int _K )
{
  K = _K;
}

/* desctructor */
Mihasher::~Mihasher()
{
  delete[] xornum;
  delete[] H;
}

/* populate tables */
void Mihasher::populate( cv::Mat & _codes, UINT32 _N, int dim1codes )
{
  N = _N;
  codes = _codes;
  UINT64 * chunks = new UINT64[m];

  UINT8 * pcodes = codes.ptr();
  for ( UINT64 i = 0; i < N; i++, pcodes += dim1codes )
  {
    split( chunks, pcodes, m, mplus, b );

    for ( int k = 0; k < m; k++ )
      H[k].insert( chunks[k], (UINT32) i );

    if( i % (int) ceil( N / 1000.0 ) == 0 )
      fflush( stdout );
  }

  delete[] chunks;
}

