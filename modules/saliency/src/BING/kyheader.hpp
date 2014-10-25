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
 // Copyright (C) 2014, OpenCV Foundation, all rights reserved.
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

#ifndef KYHEADER_H
#define KYHEADER_H

#include <assert.h>
#include <string>
#include <vector>
#include <functional>
#include <list>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <time.h>
#include <fstream>
#include <stdint.h>

// TODO: reference additional headers your program requires here

#include "opencv2/core.hpp"

#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif

#ifdef WIN32
/* windows stuff */
#else
typedef unsigned long DWORD;
typedef unsigned short WORD;
typedef unsigned int UNINT32;
typedef bool BOOL;
typedef void *HANDLE;
#endif

#ifndef _MSC_VER
typedef unsigned char BYTE;
#else
#include <windows.h>
#endif

typedef std::vector<int> vecI;
typedef const std::string CStr;
typedef const cv::Mat CMat;
typedef std::vector<std::string> vecS;
typedef std::vector<cv::Mat> vecM;
typedef std::vector<float> vecF;
typedef std::vector<double> vecD;

namespace cv
{
namespace saliency
{

enum
{
  CV_FLIP_BOTH = -1,
  CV_FLIP_VERTICAL = 0,
  CV_FLIP_HORIZONTAL = 1
};
#define CHK_IND(p) ((p).x >= 0 && (p).x < _w && (p).y >= 0 && (p).y < _h)
#define CV_Assert_(expr, args) \
{\
    if(!(expr)) {\
    String msg = cv::format args; \
    printf("%s in %s:%d\n", msg.c_str(), __FILE__, __LINE__); \
    cv::error(cv::Exception(CV_StsAssert, msg, __FUNCTION__, __FILE__, __LINE__) ); }\
}


// Return -1 if not in the list
template<typename T>
static inline int findFromList( const T &word, const std::vector<T> &strList )
{

  std::vector<cv::String>::iterator it = std::find( strList.begin(), strList.end(), word );
  if( it == strList.end() )
  {
    return -1;
  }
  else
  {
    int index = it - strList.begin();
    return index;
  }
}

template<typename T> inline T sqr( T x )
{
  return x * x;
}  // out of range risk for T = byte, ...
template<class T, int D> inline T vecSqrDist( const cv::Vec<T, D> &v1, const cv::Vec<T, D> &v2 )
{
  T s = 0;
  for ( int i = 0; i < D; i++ )
    s += sqr( v1[i] - v2[i] );
  return s;
}  // out of range risk for T = byte, ...
template<class T, int D> inline T vecDist( const cv::Vec<T, D> &v1, const cv::Vec<T, D> &v2 )
{
  return sqrt( vecSqrDist( v1, v2 ) );
}  // out of range risk for T = byte, ...

inline cv::Rect Vec4i2Rect( cv::Vec4i &v )
{
  return cv::Rect( cv::Point( v[0] - 1, v[1] - 1 ), cv::Point( v[2], v[3] ) );
}


#if defined(_MSC_VER)
# include <intrin.h>
# define POPCNT(x) __popcnt(x)
# define POPCNT64(x) (__popcnt((unsigned)(x)) + __popcnt((unsigned)((uint64_t)(x) >> 32)))
#endif

#if defined(__GNUC__)
# define POPCNT(x) __builtin_popcount(x)
# define POPCNT64(x) __builtin_popcountll(x)
#endif

inline int popcnt64( register uint64_t u )
{
  u = ( u & 0x5555555555555555 ) + ( ( u >> 1 ) & 0x5555555555555555 );
  u = ( u & 0x3333333333333333 ) + ( ( u >> 2 ) & 0x3333333333333333 );
  u = ( u & 0x0f0f0f0f0f0f0f0f ) + ( ( u >> 4 ) & 0x0f0f0f0f0f0f0f0f );
  u = ( u & 0x00ff00ff00ff00ff ) + ( ( u >> 8 ) & 0x00ff00ff00ff00ff );
  u = ( u & 0x0000ffff0000ffff ) + ( ( u >> 16 ) & 0x0000ffff0000ffff );
  u = ( u & 0x00000000ffffffff ) + ( ( u >> 32 ) & 0x00000000ffffffff );
  return (int)u;
}

inline int popcnt( register uint32_t u )
{
  u = ( u & 0x55555555 ) + ( ( u >> 1 ) & 0x55555555 );
  u = ( u & 0x33333333 ) + ( ( u >> 2 ) & 0x33333333 );
  u = ( u & 0x0f0f0f0f ) + ( ( u >> 4 ) & 0x0f0f0f0f );
  u = ( u & 0x00ff00ff ) + ( ( u >> 8 ) & 0x00ff00ff );
  u = ( u & 0x0000ffff ) + ( ( u >> 16 ) & 0x0000ffff );
  return (int)u;
}

inline int popcnt64_nibble( register uint64_t u )
{
  static const uint8_t Table[] =
  { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 };

  int c = 0;
  while ( u )
  {
    c += Table[u & 0xf];
    u >>= 4;
  }
  return (int)c;
}

inline int popcnt_nibble( register uint32_t u )
{
  static const uint8_t Table[] =
  { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 };

  int c = 0;
  while ( u )
  {
    c += Table[u & 0xf];
    u >>= 4;
  }
  return (int)c;
}

inline int popcnt64_byte( register uint64_t u )
{
#define B2(k) k, k+1, k+1, k+2
#define B4(k) B2(k), B2(k+1), B2(k+1), B2(k+2)
#define B6(k) B4(k), B4(k+1), B4(k+1), B4(k+2)
  static const uint8_t Table[] =
  { B6( 0 ), B6( 1 ), B6( 1 ), B6( 2 ) };
#undef B6
#undef B4
#undef B2

  int c = 0;
  while ( u )
  {
    c += Table[u & 0xff];
    u >>= 8;
  }
  return (int)c;
}

inline int popcnt_byte( register uint32_t u )
{
#define B2(k) k, k+1, k+1, k+2
#define B4(k) B2(k), B2(k+1), B2(k+1), B2(k+2)
#define B6(k) B4(k), B4(k+1), B4(k+1), B4(k+2)
  static const uint8_t Table[] =
  { B6( 0 ), B6( 1 ), B6( 1 ), B6( 2 ) };
#undef B6
#undef B4
#undef B2

  int c = 0;
  while ( u )
  {
    c += Table[u & 0xff];
    u >>= 8;
  }
  return (int)c;
}

}
}

#endif // KYHEADER_H
