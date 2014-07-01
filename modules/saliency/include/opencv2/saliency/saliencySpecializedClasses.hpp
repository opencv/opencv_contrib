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

/*!
 Self-Balanced Sensitivity segmenTER (SuBSENSE) foreground-background segmentation algorithm.

 Note: both grayscale and RGB/BGR images may be used with this extractor (parameters are adjusted automatically).
 For optimal grayscale results, use CV_8UC1 frames instead of CV_8UC3.

 For more details on the different parametersor on the algorithm itself, see P.-L. St-Charles et al.,
 "Flexible Background Subtraction With Self-Balanced Local Sensitivity", in CVPRW 2014.

 This algorithm is currently NOT thread-safe.
 */

class CV_EXPORTS_W MotionSaliencySuBSENSE : public BackgroundSubtractorLBSP, public MotionSaliency
{
 public:

  //! full constructor
  MotionSaliencySuBSENSE( float fRelLBSPThreshold = BGSSUBSENSE_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD, size_t nMinDescDistThreshold =
  BGSSUBSENSE_DEFAULT_DESC_DIST_THRESHOLD,
                          size_t nMinColorDistThreshold = BGSSUBSENSE_DEFAULT_COLOR_DIST_THRESHOLD, size_t nBGSamples =
                          BGSSUBSENSE_DEFAULT_NB_BG_SAMPLES,
                          size_t nRequiredBGSamples = BGSSUBSENSE_DEFAULT_REQUIRED_NB_BG_SAMPLES, size_t nSamplesForMovingAvgs =
                          BGSSUBSENSE_DEFAULT_N_SAMPLES_FOR_MV_AVGS );

  virtual ~MotionSaliencySuBSENSE();
  //! (re)initiaization method; needs to be called before starting background subtraction (note: also reinitializes the keypoints vector)
  virtual void initialize( const Mat& oInitImg, const std::vector<KeyPoint>& voKeyPoints );
  //! refreshes all samples based on the last analyzed frame
  virtual void refreshModel( float fSamplesRefreshFrac );
  //! primary model update function; the learning param is used to override the internal learning thresholds (ignored when <= 0)
  virtual void operator()( InputArray image, OutputArray fgmask, double learningRateOverride = 0 );
  //! returns a copy of the latest reconstructed background image
  void getBackgroundImage( OutputArray backgroundImage ) const;
  //! turns automatic model reset on or off
  void setAutomaticModelReset( bool );

  void read( const FileNode& fn );
  void write( FileStorage& fs ) const;

 protected:
  bool computeSaliencyImpl( const InputArray src, OutputArray dst );
  AlgorithmInfo* info() const;

  //! indicates whether internal structures have already been initialized (LBSP lookup tables, samples, etc.)
  bool m_bInitializedInternalStructs;
  //! absolute minimal color distance threshold ('R' or 'radius' in the original ViBe paper, used as the default/initial 'R(x)' value here, paired with BackgroundSubtractorLBSP::m_nDescDistThreshold)
  const size_t m_nMinColorDistThreshold;
  //! number of different samples per pixel/block to be taken from input frames to build the background model (same as 'N' in ViBe/PBAS)
  const size_t m_nBGSamples;
  //! number of similar samples needed to consider the current pixel/block as 'background' (same as '#_min' in ViBe/PBAS)
  const size_t m_nRequiredBGSamples;
  //! number of samples to use to compute the learning rate of moving averages
  const size_t m_nSamplesForMovingAvgs;
  //! current frame index, frame count since last model reset & model reset cooldown counters
  size_t m_nFrameIndex, m_nFramesSinceLastReset, m_nModelResetCooldown;
  //! last calculated non-zero desc ratio
  float m_fLastNonZeroDescRatio;
  //! specifies whether automatic model reset is enabled or not
  bool m_bAutoModelResetEnabled;
  //! specifies whether Tmin/Tmax scaling is enabled or not
  bool m_bLearningRateScalingEnabled;
  //! current learning rate caps
  float m_fCurrLearningRateLowerCap, m_fCurrLearningRateUpperCap;
  //! current kernel size for median blur post-proc filtering
  int m_nMedianBlurKernelSize;
  //! specifies the px update spread range
  bool m_bUse3x3Spread;
  //! specifies the downsampled frame size (used for cam motion analysis)
  Size m_oDownSampledFrameSize;

  //! background model pixel color intensity samples (equivalent to 'B(x)' in PBAS, but also paired with BackgroundSubtractorLBSP::m_voBGDescSamples to create our complete model)
  std::vector<Mat> m_voBGColorSamples;

  //! per-pixel update rates ('T(x)' in PBAS, which contains pixel-level 'sigmas', as referred to in ViBe)
  Mat m_oUpdateRateFrame;
  //! per-pixel distance thresholds (equivalent to 'R(x)' in PBAS, but used as a relative value to determine both intensity and descriptor variation thresholds)
  Mat m_oDistThresholdFrame;
  //! per-pixel distance variation modulators ('v(x)', relative value used to modulate 'R(x)' and 'T(x)' variations)
  Mat m_oVariationModulatorFrame;
  //! per-pixel mean distances between consecutive frames ('D_last(x)', used to detect ghosts and high variation regions in the sequence)
  Mat m_oMeanLastDistFrame;
  //! per-pixel mean minimal distances from the model ('D_min(x)' in PBAS, used to control variation magnitude and direction of 'T(x)' and 'R(x)')
  Mat m_oMeanMinDistFrame_LT, m_oMeanMinDistFrame_ST;
  //! per-pixel mean downsampled distances between consecutive frames (used to analyze camera movement and control max learning rates globally)
  Mat m_oMeanDownSampledLastDistFrame_LT, m_oMeanDownSampledLastDistFrame_ST;
  //! per-pixel mean raw segmentation results
  Mat m_oMeanRawSegmResFrame_LT, m_oMeanRawSegmResFrame_ST;
  //! per-pixel mean final segmentation results
  Mat m_oMeanFinalSegmResFrame_LT, m_oMeanFinalSegmResFrame_ST;
  //! a lookup map used to keep track of unstable regions (based on segm. noise & local dist. thresholds)
  Mat m_oUnstableRegionMask;
  //! per-pixel blink detection results ('Z(x)')
  Mat m_oBlinksFrame;
  //! pre-allocated matrix used to downsample (1/8) the input frame when needed
  Mat m_oDownSampledColorFrame;
  //! copy of previously used pixel intensities used to calculate 'D_last(x)'
  Mat m_oLastColorFrame;
  //! copy of previously used descriptors used to calculate 'D_last(x)'
  Mat m_oLastDescFrame;
  //! the foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
  Mat m_oRawFGMask_last;
  //! the foreground mask generated by the method at [t-1] (with post-proc)
  Mat m_oFGMask_last;

  //! pre-allocated CV_8UC1 matrices used to speed up morph ops
  Mat m_oFGMask_PreFlood;
  Mat m_oFGMask_FloodedHoles;
  Mat m_oFGMask_last_dilated;
  Mat m_oFGMask_last_dilated_inverted;
  Mat m_oRawFGBlinkMask_curr;
  Mat m_oRawFGBlinkMask_last;

};

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
