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
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_SALIENCY_SPECIALIZED_CLASSES_HPP__
#define __OPENCV_SALIENCY_SPECIALIZED_CLASSES_HPP_

#include "saliencyBaseClasses.hpp"
#include "BING/kyheader.h"
#include "BING/ValStructVec.h"
#include "BING/FilterTIG.h"
#include "SuBSENSE/BackgroundSubtractorLBSP.h"
#include <cstdio>
#include <string>
#include <iostream>

//! defines the default value for BackgroundSubtractorLBSP::m_fRelLBSPThreshold
#define BGSSUBSENSE_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD (0.333f)
//! defines the default value for BackgroundSubtractorLBSP::m_nDescDistThreshold
#define BGSSUBSENSE_DEFAULT_DESC_DIST_THRESHOLD (3)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nMinColorDistThreshold
#define BGSSUBSENSE_DEFAULT_COLOR_DIST_THRESHOLD (30)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nBGSamples
#define BGSSUBSENSE_DEFAULT_NB_BG_SAMPLES (50)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nRequiredBGSamples
#define BGSSUBSENSE_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nSamplesForMovingAvgs
#define BGSSUBSENSE_DEFAULT_N_SAMPLES_FOR_MV_AVGS (25)

namespace cv
{

/************************************ Specific Static Saliency Specialized Classes ************************************/

/**
 * \brief Saliency based on algorithms described in [1]
 * [1]Hou, Xiaodi, and Liqing Zhang. "Saliency detection: A spectral residual approach." Computer Vision and Pattern Recognition, 2007. CVPR'07. IEEE Conference on. IEEE, 2007.
 */
class CV_EXPORTS_W StaticSaliencySpectralResidual : public StaticSaliency
{
 public:

  //StaticSaliencySpectralResidual( const StaticSaliencySpectralResidual::Params &parameters = StaticSaliencySpectralResidual::Params() );
  StaticSaliencySpectralResidual();
  ~StaticSaliencySpectralResidual();

  typedef Ptr<Size> (Algorithm::*SizeGetter)();
  typedef void (Algorithm::*SizeSetter)( const Ptr<Size> & );

  Ptr<Size> getWsize();
  void setWsize( const Ptr<Size> &arrPtr );

  void read( const FileNode& fn );
  void write( FileStorage& fs ) const;

 protected:
  bool computeSaliencyImpl( const InputArray src, OutputArray dst );
  AlgorithmInfo* info() const;CV_PROP_RW
  Ptr<Size> resizedImageSize;

};

/************************************ Specific Motion Saliency Specialized Classes ************************************/





/************************************ Specific Objectness Specialized Classes ************************************/

/**
 * \brief Objectness algorithms based on [4]
 * [4] Cheng, Ming-Ming, et al. "BING: Binarized normed gradients for objectness estimation at 300fps." IEEE CVPR. 2014.
 */
class CV_EXPORTS_W ObjectnessBING : public Objectness
{
 public:

  ObjectnessBING();
  ~ObjectnessBING();

  void read();
  void write() const;

  // Load trained model.
  int loadTrainedModel( std::string modelName = "" );  // Return -1, 0, or 1 if partial, none, or all loaded

  // Get potential bounding boxes, each of which is represented by a Vec4i for (minX, minY, maxX, maxY).
  // The trained model should be prepared before calling this function: loadTrainedModel() or trainStageI() + trainStageII().
  // Use numDet to control the final number of proposed bounding boxes, and number of per size (scale and aspect ratio)
  void getObjBndBoxes( CMat &img3u, ValStructVec<float, Vec4i> &valBoxes, int numDetPerSize = 120 );
  void getObjBndBoxesForSingleImage( Mat img, ValStructVec<float, Vec4i> &boxes, int numDetPerSize );
  vector<float> getobjectnessValues();

  void setColorSpace( int clr = MAXBGR );
  void setTrainingPath( string trainingPath );
  void setBBResDir( string resultsDir );

  // Read matrix from binary file
  static bool matRead( const std::string& filename, Mat& M );

  enum
  {
    MAXBGR,
    HSV,
    G
  };

  inline static float LoG( float x, float y, float delta )
  {
    float d = - ( x * x + y * y ) / ( 2 * delta * delta );
    return -1.0f / ( (float) ( CV_PI ) * pow( delta, 4 ) ) * ( 1 + d ) * exp( d );
  }  // Laplacian of Gaussian

 protected:
  bool computeSaliencyImpl( const InputArray src, OutputArray dst );
  AlgorithmInfo* info() const;

 private:
  // Parameters
  double _base, _logBase;  // base for window size quantization
  int _W;  // As described in the paper: #Size, Size(_W, _H) of feature window.
  int _NSS;  // Size for non-maximal suppress
  int _maxT, _minT, _numT;  // The minimal and maximal dimensions of the template

  int _Clr;  //
  static const char* _clrName[3];

  // Names and paths to read model and to store results
  std::string _modelName, _bbResDir, _trainingPath, _resultsDir;

  vecI _svmSzIdxs;  // Indexes of active size. It's equal to _svmFilters.size() and _svmReW1f.rows
  Mat _svmFilter;  // Filters learned at stage I, each is a _H by _W CV_32F matrix
  FilterTIG _tigF;  // TIG filter
  Mat _svmReW1f;  // Re-weight parameters learned at stage II.

  // List of the rectangles' objectness value, in the same order as
  // the  vector<Vec4i> objectnessBoundingBox returned by the algorithm (in computeSaliencyImpl function)
  vector<float> objectnessValues;
  //vector<Vec4i> objectnessBoundingBox;

 private:
  // Help functions

  bool filtersLoaded()
  {
    int n = _svmSzIdxs.size();
    return n > 0 && _svmReW1f.size() == Size( 2, n ) && _svmFilter.size() == Size( _W, _W );
  }
  void predictBBoxSI( CMat &mag3u, ValStructVec<float, Vec4i> &valBoxes, vecI &sz, int NUM_WIN_PSZ = 100, bool fast = true );
  void predictBBoxSII( ValStructVec<float, Vec4i> &valBoxes, const vecI &sz );

  // Calculate the image gradient: center option as in VLFeat
  void gradientMag( CMat &imgBGR3u, Mat &mag1u );

  static void gradientRGB( CMat &bgr3u, Mat &mag1u );
  static void gradientGray( CMat &bgr3u, Mat &mag1u );
  static void gradientHSV( CMat &bgr3u, Mat &mag1u );
  static void gradientXY( CMat &x1i, CMat &y1i, Mat &mag1u );

  static inline int bgrMaxDist( const Vec3b &u, const Vec3b &v )
  {
    int b = abs( u[0] - v[0] ), g = abs( u[1] - v[1] ), r = abs( u[2] - v[2] );
    b = max( b, g );
    return max( b, r );
  }
  static inline int vecDist3b( const Vec3b &u, const Vec3b &v )
  {
    return abs( u[0] - v[0] ) + abs( u[1] - v[1] ) + abs( u[2] - v[2] );
  }

  //Non-maximal suppress
  static void nonMaxSup( CMat &matchCost1f, ValStructVec<float, Point> &matchCost, int NSS = 1, int maxPoint = 50, bool fast = true );

};

} /* namespace cv */

#endif
