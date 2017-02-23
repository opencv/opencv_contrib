/*
Copyright (c) <2014> SMHasher

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef __OPENCV_HASH_MURMUR64_HPP_
#define __OPENCV_HASH_MURMUR64_HPP_

//-----------------------------------------------------------------------------
// Block read - if your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here

FORCE_INLINE uint getblock ( const uint * p, int i )
{
  return p[i];
}

//----------
// Finalization mix - force all bits of a hash block to avalanche

// avalanches all bits to within 0.25% bias

FORCE_INLINE uint fmix32 ( uint h )
{
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  return h;
}

//-----------------------------------------------------------------------------

FORCE_INLINE void bmix32 ( uint & h1, uint & k1, uint & c1, uint & c2 )
{
  k1 *= c1;
  k1  = ROTL32(k1,11);
  k1 *= c2;
  h1 ^= k1;

  h1 = h1*3+0x52dce729;

  c1 = c1*5+0x7b7d159c;
  c2 = c2*5+0x6bce6396;
}

//-----------------------------------------------------------------------------

FORCE_INLINE void bmix32 ( uint & h1, uint & h2, uint & k1, uint & k2, uint & c1, uint & c2 )
{
  k1 *= c1;
  k1  = ROTL32(k1,11);
  k1 *= c2;
  h1 ^= k1;
  h1 += h2;

  h2 = ROTL32(h2,17);

  k2 *= c2;
  k2  = ROTL32(k2,11);
  k2 *= c1;
  h2 ^= k2;
  h2 += h1;

  h1 = h1*3+0x52dce729;
  h2 = h2*3+0x38495ab5;

  c1 = c1*5+0x7b7d159c;
  c2 = c2*5+0x6bce6396;
}

//----------

FORCE_INLINE void hashMurmurx64 ( const void * key, const int len, const uint seed, void * out )
{
  const uchar * data = (const uchar*)key;
  const int nblocks = len / 8;

  uint h1 = 0x8de1c3ac ^ seed;
  uint h2 = 0xbab98226 ^ seed;

  uint c1 = 0x95543787;
  uint c2 = 0x2ad7eb25;

  //----------
  // body

  const uint * blocks = (const uint *)(data + nblocks*8);

  for (int i = -nblocks; i; i++)
  {
    uint k1 = getblock(blocks,i*2+0);
    uint k2 = getblock(blocks,i*2+1);

    bmix32(h1,h2,k1,k2,c1,c2);
  }

  //----------
  // tail

  const uchar * tail = (const uchar*)(data + nblocks*8);

  uint k1 = 0;
  uint k2 = 0;

  switch (len & 7)
  {
  case 7:
    k2 ^= tail[6] << 16;
  case 6:
    k2 ^= tail[5] << 8;
  case 5:
    k2 ^= tail[4] << 0;
  case 4:
    k1 ^= tail[3] << 24;
  case 3:
    k1 ^= tail[2] << 16;
  case 2:
    k1 ^= tail[1] << 8;
  case 1:
    k1 ^= tail[0] << 0;
    bmix32(h1,h2,k1,k2,c1,c2);
  };

  //----------
  // finalization

  h2 ^= len;

  h1 += h2;
  h2 += h1;

  h1 = fmix32(h1);
  h2 = fmix32(h2);

  h1 += h2;
  h2 += h1;

  ((uint*)out)[0] = h1;
  ((uint*)out)[1] = h2;
}


#endif
