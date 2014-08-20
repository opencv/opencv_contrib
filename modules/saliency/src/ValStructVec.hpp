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

#ifndef __OPENCV_VAL_STRUCT_VEC_HPP__
#define __OPENCV_VAL_STRUCT_VEC_HPP__

/************************************************************************/
/* A value struct vector that supports efficient sorting                */
/************************************************************************/

namespace cv
{
namespace saliency
{

template<typename VT, typename ST>
struct ValStructVec
{
  ValStructVec()
  {
    clear();
  }
  inline int size() const
  {
    return sz;
  }
  inline void clear()
  {
    sz = 0;
    structVals.clear();
    valIdxes.clear();
  }
  inline void reserve( int resSz )
  {
    clear();
    structVals.reserve( resSz );
    valIdxes.reserve( resSz );
  }
  inline void pushBack( const VT& val, const ST& structVal )
  {
    valIdxes.push_back( std::make_pair( val, sz ) );
    structVals.push_back( structVal );
    sz++;
  }

  inline const VT& operator ()( int i ) const
  {
    return valIdxes[i].first;
  }  // Should be called after sort
  inline const ST& operator []( int i ) const
  {
    return structVals[valIdxes[i].second];
  }  // Should be called after sort
  inline VT& operator ()( int i )
  {
    return valIdxes[i].first;
  }  // Should be called after sort
  inline ST& operator []( int i )
  {
    return structVals[valIdxes[i].second];
  }  // Should be called after sort

  void sort( bool descendOrder = true );
  const std::vector<ST> &getSortedStructVal();
  std::vector<std::pair<VT, int> > getvalIdxes();
  void append( const ValStructVec<VT, ST> &newVals, int startV = 0 );

  std::vector<ST> structVals;  // struct values

 private:
  int sz;  // size of the value struct vector
  std::vector<std::pair<VT, int> > valIdxes;  // Indexes after sort
  bool smaller()
  {
    return true;
  }
  ;
  std::vector<ST> sortedStructVals;
};

template<typename VT, typename ST>
void ValStructVec<VT, ST>::append( const ValStructVec<VT, ST> &newVals, int startV )
{
  int newValsSize = newVals.size();
  for ( int i = 0; i < newValsSize; i++ )
    pushBack( (float) ( ( i + 300 ) * startV ), newVals[i] );
}

template<typename VT, typename ST>
void ValStructVec<VT, ST>::sort( bool descendOrder /* = true */)
{
  if( descendOrder )
    std::sort( valIdxes.begin(), valIdxes.end(), std::greater<std::pair<VT, int> >() );
  else
    std::sort( valIdxes.begin(), valIdxes.end(), std::less<std::pair<VT, int> >() );
}

template<typename VT, typename ST>
const std::vector<ST>& ValStructVec<VT, ST>::getSortedStructVal()
{
  sortedStructVals.resize( sz );
  for ( int i = 0; i < sz; i++ )
    sortedStructVals[i] = structVals[valIdxes[i].second];
  return sortedStructVals;
}

template<typename VT, typename ST>
std::vector<std::pair<VT, int> > ValStructVec<VT, ST>::getvalIdxes()
{
  return valIdxes;
}

} /* namespace saliency */
} /* namespace cv */

#endif //__OPENCV_VAL_STRUCT_VEC_HPP__
