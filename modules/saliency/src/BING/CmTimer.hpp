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

#ifndef __OPENCV_CM_TIMER_HPP__
#define __OPENCV_CM_TIMER_HPP__

#include "kyheader.hpp"

namespace cv
{
namespace saliency
{

class CmTimer
{
 public:
  CmTimer( CStr t ) :
      title( t )
  {
    is_started = false;
    start_clock = 0;
    cumulative_clock = 0;
    n_starts = 0;
  }

  ~CmTimer()
  {
    if( is_started )
      printf( "CmTimer '%s' is started and is being destroyed.\n", title.c_str() );
  }

  inline void Start();
  inline void Stop();
  inline void Reset();

  inline bool Report();
  inline bool StopAndReport()
  {
    Stop();
    return Report();
  }
  inline float TimeInSeconds();

 private:
  CStr title;

  CmTimer& operator=( const CmTimer& );

  bool is_started;
  clock_t start_clock;
  clock_t cumulative_clock;
  unsigned int n_starts;
};

/************************************************************************/
/*                       Implementations                                */
/************************************************************************/

void CmTimer::Start()
{
  if( is_started )
  {
    printf( "CmTimer '%s' is already started. Nothing done.\n", title.c_str() );
    return;
  }

  is_started = true;
  n_starts++;
  start_clock = clock();
}

void CmTimer::Stop()
{
  if( !is_started )
  {
    printf( "CmTimer '%s' is started. Nothing done\n", title.c_str() );
    return;
  }

  cumulative_clock += clock() - start_clock;
  is_started = false;
}

void CmTimer::Reset()
{
  if( is_started )
  {
    printf( "CmTimer '%s'is started during reset request.\n Only reset cumulative time.\n", title.c_str() );
    return;
  }
  cumulative_clock = 0;
}

bool CmTimer::Report()
{
  if( is_started )
  {
    printf( "CmTimer '%s' is started.\n Cannot provide a time report.", title.c_str() );
    return false;
  }

  float timeUsed = TimeInSeconds();
  printf( "[%s] CumuTime: %gs, #run: %d, AvgTime: %gs\n", title.c_str(), timeUsed, n_starts, timeUsed / n_starts );
  return true;
}

float CmTimer::TimeInSeconds()
{
  if( is_started )
  {
    printf( "CmTimer '%s' is started. Nothing done\n", title.c_str() );
    return 0;
  }
  return float( cumulative_clock ) / CLOCKS_PER_SEC;
}

}  // namespace saliency
}  // namespace cv

#endif // __OPENCV_CM_TIMER_HPP__
