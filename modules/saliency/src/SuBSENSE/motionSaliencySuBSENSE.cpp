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

#include "precomp.hpp"

#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iomanip>

/*
 *
 * Intrinsic parameters for our method are defined here; tuning these for better
 * performance should not be required in most cases -- although improvements in
 * very specific scenarios are always possible.
 *
 * Note that the current configuration was used to obtain the results presented
 * in our paper, in conjunction with the 2014 CVPRW on Change Detection.
 *
 */

//! defines the threshold value(s) used to detect long-term ghosting and trigger the fast edge-based absorption heuristic
#define GHOSTDET_D_MAX (0.010f) // defines 'negligible' change here
#define GHOSTDET_S_MIN (0.995f) // defines the required minimum local foreground saturation value
//! parameter used to scale dynamic distance threshold adjustments ('R(x)')
#define FEEDBACK_R_VAR (0.01f)
//! parameters used to adjust the variation step size of 'v(x)'
#define FEEDBACK_V_INCR  (1.000f)
#define FEEDBACK_V_DECR  (0.100f)
//! parameters used to scale dynamic learning rate adjustments  ('T(x)')
#define FEEDBACK_T_DECR  (0.2500f)
#define FEEDBACK_T_INCR  (0.5000f)
#define FEEDBACK_T_LOWER (2.0000f)
#define FEEDBACK_T_UPPER (256.00f)
//! parameters used to define 'unstable' regions, based on segm noise/bg dynamics and local dist threshold values
#define UNSTABLE_REG_RATIO_MIN 0.100f
#define UNSTABLE_REG_RDIST_MIN 3.000f
//! parameters used to scale the relative LBSP intensity threshold used for internal comparisons
#define LBSPDESC_NONZERO_RATIO_MIN 0.100f
#define LBSPDESC_NONZERO_RATIO_MAX 0.500f
//! parameters used to define model reset/learning rate boosts in our frame-level component
#define FRAMELEVEL_COLOR_DIFF_RESET_THRESHOLD 15
#define FRAMELEVEL_ANALYSIS_DOWNSAMPLE_RATIO 8

// local define used for debug purposes only
#define DISPLAY_SUBSENSE_DEBUG_INFO 0
// local define used to specify the default internal frame size
#define DEFAULT_FRAME_SIZE Size(320,240)
// local define used to specify the color dist threshold offset used for unstable regions
#define STAB_COLOR_DIST_OFFSET m_nMinColorDistThreshold/5
// local define used to specify the desc dist threshold offset used for unstable regions
#define UNSTAB_DESC_DIST_OFFSET m_nDescDistThreshold
// local define used to determine the median blur kernel size
#define DEFAULT_MEDIAN_BLUR_KERNEL_SIZE (9)

static const size_t s_nColorMaxDataRange_1ch = UCHAR_MAX;
static const size_t s_nDescMaxDataRange_1ch = LBSP::DESC_SIZE * 8;
static const size_t s_nColorMaxDataRange_3ch = s_nColorMaxDataRange_1ch * 3;
static const size_t s_nDescMaxDataRange_3ch = s_nDescMaxDataRange_1ch * 3;

namespace cv
{

MotionSaliencySuBSENSE::MotionSaliencySuBSENSE( float fRelLBSPThreshold, size_t nMinDescDistThreshold, size_t nMinColorDistThreshold,
                                                size_t nBGSamples, size_t nRequiredBGSamples, size_t nSamplesForMovingAvgs )

:
    BackgroundSubtractorLBSP( fRelLBSPThreshold, nMinDescDistThreshold ),
    m_bInitializedInternalStructs( false ),
    m_nMinColorDistThreshold( nMinColorDistThreshold ),
    m_nBGSamples( nBGSamples ),
    m_nRequiredBGSamples( nRequiredBGSamples ),
    m_nSamplesForMovingAvgs( nSamplesForMovingAvgs ),
    m_nFrameIndex( SIZE_MAX ),
    m_nFramesSinceLastReset( 0 ),
    m_nModelResetCooldown( 0 ),
    m_fLastNonZeroDescRatio( 0.0f ),
    m_bAutoModelResetEnabled( true ),
    m_bLearningRateScalingEnabled( true ),
    m_fCurrLearningRateLowerCap( FEEDBACK_T_LOWER ),
    m_fCurrLearningRateUpperCap( FEEDBACK_T_UPPER ),
    m_nMedianBlurKernelSize( DEFAULT_MEDIAN_BLUR_KERNEL_SIZE ),
    m_bUse3x3Spread( true )
{
  CV_Assert( m_nBGSamples > 0 && m_nRequiredBGSamples <= m_nBGSamples );
  CV_Assert( m_nMinColorDistThreshold>=STAB_COLOR_DIST_OFFSET );
  className = "PBAS";
}

MotionSaliencySuBSENSE::~MotionSaliencySuBSENSE() {}


void MotionSaliencySuBSENSE::read( const cv::FileNode& /*fn*/ )
{
  //params.read( fn );
}

void MotionSaliencySuBSENSE::write( cv::FileStorage& /*fs*/ ) const
{
  //params.write( fs );
}

bool MotionSaliencySuBSENSE::computeSaliencyImpl( const InputArray /*src*/, OutputArray /*dst*/)
{

  return true;
}

void MotionSaliencySuBSENSE::initialize( const Mat& oInitImg, const std::vector<KeyPoint>& voKeyPoints )
{
  // == init
  CV_Assert( !oInitImg.empty() && oInitImg.cols > 0 && oInitImg.rows > 0 );
  CV_Assert( oInitImg.type()==CV_8UC3 || oInitImg.type()==CV_8UC1 );
  if( oInitImg.type() == CV_8UC3 )
  {
    std::vector<Mat> voInitImgChannels;
    split( oInitImg, voInitImgChannels );
   /* bool eq = std::equal( voInitImgChannels[0].begin<uchar>(), voInitImgChannels[0].end<uchar>(), voInitImgChannels[1].begin<uchar>() )
        && std::equal( voInitImgChannels[1].begin<uchar>(), voInitImgChannels[1].end<uchar>(), voInitImgChannels[2].begin<uchar>() );
    if( eq )
      std::cout << std::endl
                << "\tMotionSaliencySuBSENSE : Warning, grayscale images should always be passed in CV_8UC1 format for optimal performance."
                << std::endl; */
  }
  std::vector<KeyPoint> voNewKeyPoints;
  if( voKeyPoints.empty() )
  {
    DenseFeatureDetector oKPDDetector( 1.f, 1, 1.f, 1, 0, true, false );
    voNewKeyPoints.reserve( oInitImg.rows * oInitImg.cols );
    oKPDDetector.detect( Mat( oInitImg.size(), oInitImg.type() ), voNewKeyPoints );
  }
  else
    voNewKeyPoints = voKeyPoints;
  const size_t nOrigKeyPointsCount = voNewKeyPoints.size();
  CV_Assert( nOrigKeyPointsCount > 0 );
  LBSP::validateKeyPoints( voNewKeyPoints, oInitImg.size() );
  CV_Assert( !voNewKeyPoints.empty() );
  m_voKeyPoints = voNewKeyPoints;
  m_nKeyPoints = m_voKeyPoints.size();
  m_oImgSize = oInitImg.size();
  m_nImgType = oInitImg.type();
  m_nImgChannels = oInitImg.channels();
  m_nFrameIndex = 0;
  m_nFramesSinceLastReset = 0;
  m_nModelResetCooldown = 0;
  m_fLastNonZeroDescRatio = 0.0f;
  const int nTotImgPixels = m_oImgSize.height * m_oImgSize.width;
  if( (int) nOrigKeyPointsCount >= nTotImgPixels / 2 && nTotImgPixels >= DEFAULT_FRAME_SIZE.area() )
  {
    m_bLearningRateScalingEnabled = true;
    m_bAutoModelResetEnabled = true;
    m_bUse3x3Spread = ! ( nTotImgPixels > DEFAULT_FRAME_SIZE.area() * 2 );
    const int nRawMedianBlurKernelSize = std::min(
        (int) floor( (float) nTotImgPixels / DEFAULT_FRAME_SIZE.area() + 0.5f ) + DEFAULT_MEDIAN_BLUR_KERNEL_SIZE, 14 );
    m_nMedianBlurKernelSize = ( nRawMedianBlurKernelSize % 2 ) ? nRawMedianBlurKernelSize : nRawMedianBlurKernelSize - 1;
    m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER;
    m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER;
  }
  else
  {
    m_bLearningRateScalingEnabled = false;
    m_bAutoModelResetEnabled = false;
    m_bUse3x3Spread = true;
    m_nMedianBlurKernelSize = DEFAULT_MEDIAN_BLUR_KERNEL_SIZE;
    m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER * 2;
    m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER * 2;
  }
  //std::cout << m_oImgSize << " => m_nMedianBlurKernelSize=" << m_nMedianBlurKernelSize << ", with 3x3Spread=" << m_bUse3x3Spread << ", with Tscaling=" << m_bLearningRateScalingEnabled << std::endl;
  m_oUpdateRateFrame.create( m_oImgSize, CV_32FC1 );
  m_oUpdateRateFrame = Scalar( m_fCurrLearningRateLowerCap );
  m_oDistThresholdFrame.create( m_oImgSize, CV_32FC1 );
  m_oDistThresholdFrame = Scalar( 1.0f );
  m_oVariationModulatorFrame.create( m_oImgSize, CV_32FC1 );
  m_oVariationModulatorFrame = Scalar( 10.0f );  // should always be >= FEEDBACK_V_DECR
  m_oMeanLastDistFrame.create( m_oImgSize, CV_32FC1 );
  m_oMeanLastDistFrame = Scalar( 0.0f );
  m_oMeanMinDistFrame_LT.create( m_oImgSize, CV_32FC1 );
  m_oMeanMinDistFrame_LT = Scalar( 0.0f );
  m_oMeanMinDistFrame_ST.create( m_oImgSize, CV_32FC1 );
  m_oMeanMinDistFrame_ST = Scalar( 0.0f );
  m_oDownSampledFrameSize = Size( m_oImgSize.width / FRAMELEVEL_ANALYSIS_DOWNSAMPLE_RATIO, m_oImgSize.height / FRAMELEVEL_ANALYSIS_DOWNSAMPLE_RATIO );
  m_oMeanDownSampledLastDistFrame_LT.create( m_oDownSampledFrameSize, CV_32FC( (int )m_nImgChannels ) );
  m_oMeanDownSampledLastDistFrame_LT = Scalar( 0.0f );
  m_oMeanDownSampledLastDistFrame_ST.create( m_oDownSampledFrameSize, CV_32FC( (int )m_nImgChannels ) );
  m_oMeanDownSampledLastDistFrame_ST = Scalar( 0.0f );
  m_oMeanRawSegmResFrame_LT.create( m_oImgSize, CV_32FC1 );
  m_oMeanRawSegmResFrame_LT = Scalar( 0.0f );
  m_oMeanRawSegmResFrame_ST.create( m_oImgSize, CV_32FC1 );
  m_oMeanRawSegmResFrame_ST = Scalar( 0.0f );
  m_oMeanFinalSegmResFrame_LT.create( m_oImgSize, CV_32FC1 );
  m_oMeanFinalSegmResFrame_LT = Scalar( 0.0f );
  m_oMeanFinalSegmResFrame_ST.create( m_oImgSize, CV_32FC1 );
  m_oMeanFinalSegmResFrame_ST = Scalar( 0.0f );
  m_oUnstableRegionMask.create( m_oImgSize, CV_8UC1 );
  m_oUnstableRegionMask = Scalar_<uchar>( 0 );
  m_oBlinksFrame.create( m_oImgSize, CV_8UC1 );
  m_oBlinksFrame = Scalar_<uchar>( 0 );
  m_oDownSampledColorFrame.create( m_oDownSampledFrameSize, CV_8UC( (int )m_nImgChannels ) );
  m_oDownSampledColorFrame = Scalar_<uchar>::all( 0 );
  m_oLastColorFrame.create( m_oImgSize, CV_8UC( (int )m_nImgChannels ) );
  m_oLastColorFrame = Scalar_<uchar>::all( 0 );
  m_oLastDescFrame.create( m_oImgSize, CV_16UC( (int )m_nImgChannels ) );
  m_oLastDescFrame = Scalar_<ushort>::all( 0 );
  m_oRawFGMask_last.create( m_oImgSize, CV_8UC1 );
  m_oRawFGMask_last = Scalar_<uchar>( 0 );
  m_oFGMask_last.create( m_oImgSize, CV_8UC1 );
  m_oFGMask_last = Scalar_<uchar>( 0 );
  m_oFGMask_last_dilated.create( m_oImgSize, CV_8UC1 );
  m_oFGMask_last_dilated = Scalar_<uchar>( 0 );
  m_oFGMask_last_dilated_inverted.create( m_oImgSize, CV_8UC1 );
  m_oFGMask_last_dilated_inverted = Scalar_<uchar>( 0 );
  m_oFGMask_FloodedHoles.create( m_oImgSize, CV_8UC1 );
  m_oFGMask_FloodedHoles = Scalar_<uchar>( 0 );
  m_oFGMask_PreFlood.create( m_oImgSize, CV_8UC1 );
  m_oFGMask_PreFlood = Scalar_<uchar>( 0 );
  m_oRawFGBlinkMask_curr.create( m_oImgSize, CV_8UC1 );
  m_oRawFGBlinkMask_curr = Scalar_<uchar>( 0 );
  m_oRawFGBlinkMask_last.create( m_oImgSize, CV_8UC1 );
  m_oRawFGBlinkMask_last = Scalar_<uchar>( 0 );
  m_voBGColorSamples.resize( m_nBGSamples );
  m_voBGDescSamples.resize( m_nBGSamples );
  for ( size_t s = 0; s < m_nBGSamples; ++s )
  {
    m_voBGColorSamples[s].create( m_oImgSize, CV_8UC( (int )m_nImgChannels ) );
    m_voBGColorSamples[s] = Scalar_<uchar>::all( 0 );
    m_voBGDescSamples[s].create( m_oImgSize, CV_16UC( (int )m_nImgChannels ) );
    m_voBGDescSamples[s] = Scalar_<ushort>::all( 0 );
  }
  if( m_nImgChannels == 1 )
  {
    for ( size_t t = 0; t <= UCHAR_MAX; ++t )
      m_anLBSPThreshold_8bitLUT[t] = saturate_cast<uchar>( ( m_nLBSPThresholdOffset + t * m_fRelLBSPThreshold ) / 3 );
    for ( size_t k = 0; k < m_nKeyPoints; ++k )
    {
      const int y_orig = (int) m_voKeyPoints[k].pt.y;
      const int x_orig = (int) m_voKeyPoints[k].pt.x;
      CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols && m_oLastColorFrame.step.p[1]==1);
      const size_t idx_color = m_oLastColorFrame.cols * y_orig + x_orig;
      CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
      const size_t idx_desc = idx_color * 2;
      m_oLastColorFrame.data[idx_color] = oInitImg.data[idx_color];
      LBSP::computeGrayscaleDescriptor( oInitImg, oInitImg.data[idx_color], x_orig, y_orig, m_anLBSPThreshold_8bitLUT[oInitImg.data[idx_color]],
                                        * ( (ushort*) ( m_oLastDescFrame.data + idx_desc ) ) );
    }
  }
  else
  {  //m_nImgChannels==3
    for ( size_t t = 0; t <= UCHAR_MAX; ++t )
      m_anLBSPThreshold_8bitLUT[t] = saturate_cast<uchar>( m_nLBSPThresholdOffset + t * m_fRelLBSPThreshold );
    for ( size_t k = 0; k < m_nKeyPoints; ++k )
    {
      const int y_orig = (int) m_voKeyPoints[k].pt.y;
      const int x_orig = (int) m_voKeyPoints[k].pt.x;
      CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols*3 && m_oLastColorFrame.step.p[1]==3);
      const size_t idx_color = 3 * ( m_oLastColorFrame.cols * y_orig + x_orig );
      CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
      const size_t idx_desc = idx_color * 2;
      for ( size_t c = 0; c < 3; ++c )
      {
        const uchar nCurrBGInitColor = oInitImg.data[idx_color + c];
        m_oLastColorFrame.data[idx_color + c] = nCurrBGInitColor;
        LBSP::computeSingleRGBDescriptor( oInitImg, nCurrBGInitColor, x_orig, y_orig, c, m_anLBSPThreshold_8bitLUT[nCurrBGInitColor],
                                          ( (ushort*) ( m_oLastDescFrame.data + idx_desc ) )[c] );
      }
    }
  }
  m_bInitializedInternalStructs = true;
  refreshModel( 1.0f );
  m_bInitialized = true;
}

void MotionSaliencySuBSENSE::refreshModel( float fSamplesRefreshFrac )
{
  // == refresh
  CV_Assert( m_bInitializedInternalStructs );
  CV_Assert( fSamplesRefreshFrac > 0.0f && fSamplesRefreshFrac <= 1.0f );
  const size_t nBGSamplesToRefresh = fSamplesRefreshFrac < 1.0f ? (size_t) ( fSamplesRefreshFrac * m_nBGSamples ) : m_nBGSamples;
  const size_t nRefreshStartPos = fSamplesRefreshFrac < 1.0f ? rand() % m_nBGSamples : 0;
  if( m_nImgChannels == 1 )
  {
    for ( size_t k = 0; k < m_nKeyPoints; ++k )
    {
      const int y_orig = (int) m_voKeyPoints[k].pt.y;
      const int x_orig = (int) m_voKeyPoints[k].pt.x;
      CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols && m_oLastColorFrame.step.p[1]==1);
      const size_t idx_orig_color = m_oLastColorFrame.cols * y_orig + x_orig;
      CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
      const size_t idx_orig_desc = idx_orig_color * 2;
      for ( size_t s = nRefreshStartPos; s < nRefreshStartPos + nBGSamplesToRefresh; ++s )
      {
        int y_sample, x_sample;
        getRandSamplePosition( x_sample, y_sample, x_orig, y_orig, LBSP::PATCH_SIZE / 2, m_oImgSize );
        const size_t idx_sample_color = m_oLastColorFrame.cols * y_sample + x_sample;
        const size_t idx_sample_desc = idx_sample_color * 2;
        const size_t idx_sample = s % m_nBGSamples;
        m_voBGColorSamples[idx_sample].data[idx_orig_color] = m_oLastColorFrame.data[idx_sample_color];
        * ( (ushort*) ( m_voBGDescSamples[idx_sample].data + idx_orig_desc ) ) = * ( (ushort*) ( m_oLastDescFrame.data + idx_sample_desc ) );
      }
    }
  }
  else
  {  //m_nImgChannels==3
    for ( size_t k = 0; k < m_nKeyPoints; ++k )
    {
      const int y_orig = (int) m_voKeyPoints[k].pt.y;
      const int x_orig = (int) m_voKeyPoints[k].pt.x;
      CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols*3 && m_oLastColorFrame.step.p[1]==3);
      const size_t idx_orig_color = 3 * ( m_oLastColorFrame.cols * y_orig + x_orig );
      CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
      const size_t idx_orig_desc = idx_orig_color * 2;
      for ( size_t s = nRefreshStartPos; s < nRefreshStartPos + nBGSamplesToRefresh; ++s )
      {
        int y_sample, x_sample;
        getRandSamplePosition( x_sample, y_sample, x_orig, y_orig, LBSP::PATCH_SIZE / 2, m_oImgSize );
        const size_t idx_sample_color = 3 * ( m_oLastColorFrame.cols * y_sample + x_sample );
        const size_t idx_sample_desc = idx_sample_color * 2;
        const size_t idx_sample = s % m_nBGSamples;
        uchar* bg_color_ptr = m_voBGColorSamples[idx_sample].data + idx_orig_color;
        ushort* bg_desc_ptr = (ushort*) ( m_voBGDescSamples[idx_sample].data + idx_orig_desc );
        const uchar* const init_color_ptr = m_oLastColorFrame.data + idx_sample_color;
        const ushort* const init_desc_ptr = (ushort*) ( m_oLastDescFrame.data + idx_sample_desc );
        for ( size_t c = 0; c < 3; ++c )
        {
          bg_color_ptr[c] = init_color_ptr[c];
          bg_desc_ptr[c] = init_desc_ptr[c];
        }
      }
    }
  }
}

void MotionSaliencySuBSENSE::operator()( InputArray _image, OutputArray _fgmask, double learningRateOverride )
{
  // == process
  CV_DbgAssert(m_bInitialized);
  Mat oInputImg = _image.getMat();
  CV_DbgAssert(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize);
  _fgmask.create( m_oImgSize, CV_8UC1 );
  Mat oCurrFGMask = _fgmask.getMat();
  memset( oCurrFGMask.data, 0, oCurrFGMask.cols * oCurrFGMask.rows );
  size_t nNonZeroDescCount = 0;
  const float fRollAvgFactor_LT = 1.0f / std::min( ++m_nFrameIndex, m_nSamplesForMovingAvgs * 4 );
  const float fRollAvgFactor_ST = 1.0f / std::min( m_nFrameIndex, m_nSamplesForMovingAvgs );
  if( m_nImgChannels == 1 )
  {
    for ( size_t k = 0; k < m_nKeyPoints; ++k )
    {
      const int x = (int) m_voKeyPoints[k].pt.x;
      const int y = (int) m_voKeyPoints[k].pt.y;
      const size_t idx_uchar = m_oImgSize.width * y + x;
      const size_t idx_ushrt = idx_uchar * 2;
      const size_t idx_flt32 = idx_uchar * 4;
      const uchar nCurrColor = oInputImg.data[idx_uchar];
      size_t nMinDescDist = s_nDescMaxDataRange_1ch;
      size_t nMinSumDist = s_nColorMaxDataRange_1ch;
      float* pfCurrDistThresholdFactor = (float*) ( m_oDistThresholdFrame.data + idx_flt32 );
      float* pfCurrVariationFactor = (float*) ( m_oVariationModulatorFrame.data + idx_flt32 );
      float* pfCurrLearningRate = ( (float*) ( m_oUpdateRateFrame.data + idx_flt32 ) );
      float* pfCurrMeanLastDist = ( (float*) ( m_oMeanLastDistFrame.data + idx_flt32 ) );
      float* pfCurrMeanMinDist_LT = ( (float*) ( m_oMeanMinDistFrame_LT.data + idx_flt32 ) );
      float* pfCurrMeanMinDist_ST = ( (float*) ( m_oMeanMinDistFrame_ST.data + idx_flt32 ) );
      float* pfCurrMeanRawSegmRes_LT = ( (float*) ( m_oMeanRawSegmResFrame_LT.data + idx_flt32 ) );
      float* pfCurrMeanRawSegmRes_ST = ( (float*) ( m_oMeanRawSegmResFrame_ST.data + idx_flt32 ) );
      float* pfCurrMeanFinalSegmRes_LT = ( (float*) ( m_oMeanFinalSegmResFrame_LT.data + idx_flt32 ) );
      float* pfCurrMeanFinalSegmRes_ST = ( (float*) ( m_oMeanFinalSegmResFrame_ST.data + idx_flt32 ) );
      ushort& nLastIntraDesc = * ( (ushort*) ( m_oLastDescFrame.data + idx_ushrt ) );
      uchar& nLastColor = m_oLastColorFrame.data[idx_uchar];
      const size_t nCurrColorDistThreshold = (size_t) ( ( ( *pfCurrDistThresholdFactor ) * m_nMinColorDistThreshold )
          - ( ( !m_oUnstableRegionMask.data[idx_uchar] ) * STAB_COLOR_DIST_OFFSET ) ) / 2;
      const size_t nCurrDescDistThreshold = ( (size_t) 1 << ( (size_t) floor( *pfCurrDistThresholdFactor + 0.5f ) ) ) + m_nDescDistThreshold
          + ( m_oUnstableRegionMask.data[idx_uchar] * UNSTAB_DESC_DIST_OFFSET );
      ushort nCurrInterDesc, nCurrIntraDesc;
      LBSP::computeGrayscaleDescriptor( oInputImg, nCurrColor, x, y, m_anLBSPThreshold_8bitLUT[nCurrColor], nCurrIntraDesc );
      m_oUnstableRegionMask.data[idx_uchar] =
          ( ( *pfCurrDistThresholdFactor ) > UNSTABLE_REG_RDIST_MIN
              || ( *pfCurrMeanRawSegmRes_LT - *pfCurrMeanFinalSegmRes_LT ) > UNSTABLE_REG_RATIO_MIN
              || ( *pfCurrMeanRawSegmRes_ST - *pfCurrMeanFinalSegmRes_ST ) > UNSTABLE_REG_RATIO_MIN ) ? 1 : 0;
      size_t nGoodSamplesCount = 0, nSampleIdx = 0;
      while ( nGoodSamplesCount < m_nRequiredBGSamples && nSampleIdx < m_nBGSamples )
      {
        const uchar& nBGColor = m_voBGColorSamples[nSampleIdx].data[idx_uchar];
        {
          const size_t nColorDist = absdiff_uchar( nCurrColor, nBGColor );
          if( nColorDist > nCurrColorDistThreshold )
            goto failedcheck1ch;
          const ushort& nBGIntraDesc = * ( (ushort*) ( m_voBGDescSamples[nSampleIdx].data + idx_ushrt ) );
          const size_t nIntraDescDist = hdist_ushort_8bitLUT( nCurrIntraDesc, nBGIntraDesc );
          LBSP::computeGrayscaleDescriptor( oInputImg, nBGColor, x, y, m_anLBSPThreshold_8bitLUT[nBGColor], nCurrInterDesc );
          const size_t nInterDescDist = hdist_ushort_8bitLUT( nCurrInterDesc, nBGIntraDesc );
          const size_t nDescDist = ( nIntraDescDist + nInterDescDist ) / 2;
          if( nDescDist > nCurrDescDistThreshold )
            goto failedcheck1ch;
          const size_t nSumDist = std::min( ( nDescDist / 4 ) * ( s_nColorMaxDataRange_1ch / s_nDescMaxDataRange_1ch ) + nColorDist,
                                            s_nColorMaxDataRange_1ch );
          if( nSumDist > nCurrColorDistThreshold )
            goto failedcheck1ch;
          if( nMinDescDist > nDescDist )
            nMinDescDist = nDescDist;
          if( nMinSumDist > nSumDist )
            nMinSumDist = nSumDist;
          nGoodSamplesCount++;
        }
        failedcheck1ch : nSampleIdx++;
      }
      const float fNormalizedLastDist = ( (float) absdiff_uchar( nLastColor, nCurrColor ) / s_nColorMaxDataRange_1ch
          + (float) hdist_ushort_8bitLUT( nLastIntraDesc, nCurrIntraDesc ) / s_nDescMaxDataRange_1ch ) / 2;
      *pfCurrMeanLastDist = ( *pfCurrMeanLastDist ) * ( 1.0f - fRollAvgFactor_ST ) + fNormalizedLastDist * fRollAvgFactor_ST;
      if( nGoodSamplesCount < m_nRequiredBGSamples )
      {
        // == foreground
        const float fNormalizedMinDist = std::min(
            1.0f,
            ( (float) nMinSumDist / s_nColorMaxDataRange_1ch + (float) nMinDescDist / s_nDescMaxDataRange_1ch ) / 2
                + (float) ( m_nRequiredBGSamples - nGoodSamplesCount ) / m_nRequiredBGSamples );
        *pfCurrMeanMinDist_LT = ( *pfCurrMeanMinDist_LT ) * ( 1.0f - fRollAvgFactor_LT ) + fNormalizedMinDist * fRollAvgFactor_LT;
        *pfCurrMeanMinDist_ST = ( *pfCurrMeanMinDist_ST ) * ( 1.0f - fRollAvgFactor_ST ) + fNormalizedMinDist * fRollAvgFactor_ST;
        *pfCurrMeanRawSegmRes_LT = ( *pfCurrMeanRawSegmRes_LT ) * ( 1.0f - fRollAvgFactor_LT ) + fRollAvgFactor_LT;
        *pfCurrMeanRawSegmRes_ST = ( *pfCurrMeanRawSegmRes_ST ) * ( 1.0f - fRollAvgFactor_ST ) + fRollAvgFactor_ST;
        oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
        if( m_nModelResetCooldown && ( rand() % (size_t) FEEDBACK_T_LOWER ) == 0 )
        {
          const size_t s_rand = rand() % m_nBGSamples;
          * ( (ushort*) ( m_voBGDescSamples[s_rand].data + idx_ushrt ) ) = nCurrIntraDesc;
          m_voBGColorSamples[s_rand].data[idx_uchar] = nCurrColor;
        }
      }
      else
      {
        // == background
        const float fNormalizedMinDist = ( (float) nMinSumDist / s_nColorMaxDataRange_1ch + (float) nMinDescDist / s_nDescMaxDataRange_1ch ) / 2;
        *pfCurrMeanMinDist_LT = ( *pfCurrMeanMinDist_LT ) * ( 1.0f - fRollAvgFactor_LT ) + fNormalizedMinDist * fRollAvgFactor_LT;
        *pfCurrMeanMinDist_ST = ( *pfCurrMeanMinDist_ST ) * ( 1.0f - fRollAvgFactor_ST ) + fNormalizedMinDist * fRollAvgFactor_ST;
        *pfCurrMeanRawSegmRes_LT = ( *pfCurrMeanRawSegmRes_LT ) * ( 1.0f - fRollAvgFactor_LT );
        *pfCurrMeanRawSegmRes_ST = ( *pfCurrMeanRawSegmRes_ST ) * ( 1.0f - fRollAvgFactor_ST );
        const size_t nLearningRate = learningRateOverride > 0 ? (size_t) ceil( learningRateOverride ) : (size_t) ceil( *pfCurrLearningRate );
        if( ( rand() % nLearningRate ) == 0 )
        {
          const size_t s_rand = rand() % m_nBGSamples;
          * ( (ushort*) ( m_voBGDescSamples[s_rand].data + idx_ushrt ) ) = nCurrIntraDesc;
          m_voBGColorSamples[s_rand].data[idx_uchar] = nCurrColor;
        }
        int x_rand, y_rand;
        const bool bCurrUsing3x3Spread = m_bUse3x3Spread && !m_oUnstableRegionMask.data[idx_uchar];
        if( bCurrUsing3x3Spread )
          getRandNeighborPosition_3x3( x_rand, y_rand, x, y, LBSP::PATCH_SIZE / 2, m_oImgSize );
        else
          getRandNeighborPosition_5x5( x_rand, y_rand, x, y, LBSP::PATCH_SIZE / 2, m_oImgSize );
        const size_t n_rand = rand();
        const size_t idx_rand_uchar = m_oImgSize.width * y_rand + x_rand;
        const size_t idx_rand_flt32 = idx_rand_uchar * 4;
        const float fRandMeanLastDist = * ( (float*) ( m_oMeanLastDistFrame.data + idx_rand_flt32 ) );
        const float fRandMeanRawSegmRes = * ( (float*) ( m_oMeanRawSegmResFrame_ST.data + idx_rand_flt32 ) );
        if( ( n_rand % ( bCurrUsing3x3Spread ? nLearningRate : ( nLearningRate / 2 + 1 ) ) ) == 0
            || ( fRandMeanRawSegmRes > GHOSTDET_S_MIN && fRandMeanLastDist < GHOSTDET_D_MAX
                && ( n_rand % ( (size_t) m_fCurrLearningRateLowerCap ) ) == 0 ) )
        {
          const size_t idx_rand_ushrt = idx_rand_uchar * 2;
          const size_t s_rand = rand() % m_nBGSamples;
          * ( (ushort*) ( m_voBGDescSamples[s_rand].data + idx_rand_ushrt ) ) = nCurrIntraDesc;
          m_voBGColorSamples[s_rand].data[idx_rand_uchar] = nCurrColor;
        }
      }
      if( m_oFGMask_last.data[idx_uchar]
          || ( std::min( *pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST ) < UNSTABLE_REG_RATIO_MIN && oCurrFGMask.data[idx_uchar] ) )
      {
        if( ( *pfCurrLearningRate ) < m_fCurrLearningRateUpperCap )
          *pfCurrLearningRate += FEEDBACK_T_INCR / ( std::max( *pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST ) * ( *pfCurrVariationFactor ) );
      }
      else if( ( *pfCurrLearningRate ) > m_fCurrLearningRateLowerCap )
        *pfCurrLearningRate -= FEEDBACK_T_DECR * ( *pfCurrVariationFactor ) / std::max( *pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST );
      if( ( *pfCurrLearningRate ) < m_fCurrLearningRateLowerCap )
        *pfCurrLearningRate = m_fCurrLearningRateLowerCap;
      else if( ( *pfCurrLearningRate ) > m_fCurrLearningRateUpperCap )
        *pfCurrLearningRate = m_fCurrLearningRateUpperCap;
      if( std::max( *pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST ) > UNSTABLE_REG_RATIO_MIN && m_oBlinksFrame.data[idx_uchar] )
        ( *pfCurrVariationFactor ) += FEEDBACK_V_INCR;
      else if( ( *pfCurrVariationFactor ) > FEEDBACK_V_DECR )
      {
        ( *pfCurrVariationFactor ) -= m_oFGMask_last.data[idx_uchar] ? FEEDBACK_V_DECR / 4 :
                                      m_oUnstableRegionMask.data[idx_uchar] ? FEEDBACK_V_DECR / 2 : FEEDBACK_V_DECR;
        if( ( *pfCurrVariationFactor ) < FEEDBACK_V_DECR )
          ( *pfCurrVariationFactor ) = FEEDBACK_V_DECR;
      }
      if( ( *pfCurrDistThresholdFactor ) < std::pow( 1.0f + std::min( *pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST ) * 2, 2 ) )
        ( *pfCurrDistThresholdFactor ) += FEEDBACK_R_VAR * ( *pfCurrVariationFactor - FEEDBACK_V_DECR );
      else
      {
        ( *pfCurrDistThresholdFactor ) -= FEEDBACK_R_VAR / ( *pfCurrVariationFactor );
        if( ( *pfCurrDistThresholdFactor ) < 1.0f )
          ( *pfCurrDistThresholdFactor ) = 1.0f;
      }
      if( popcount_ushort_8bitsLUT( nCurrIntraDesc ) >= 2 )
        ++nNonZeroDescCount;
      nLastIntraDesc = nCurrIntraDesc;
      nLastColor = nCurrColor;
    }
  }
  else
  {  //m_nImgChannels==3
    for ( size_t k = 0; k < m_nKeyPoints; ++k )
    {
      const int x = (int) m_voKeyPoints[k].pt.x;
      const int y = (int) m_voKeyPoints[k].pt.y;
      const size_t idx_uchar = m_oImgSize.width * y + x;
      const size_t idx_flt32 = idx_uchar * 4;
      const size_t idx_uchar_rgb = idx_uchar * 3;
      const size_t idx_ushrt_rgb = idx_uchar_rgb * 2;
      const uchar* const anCurrColor = oInputImg.data + idx_uchar_rgb;
      size_t nMinTotDescDist = s_nDescMaxDataRange_3ch;
      size_t nMinTotSumDist = s_nColorMaxDataRange_3ch;
      float* pfCurrDistThresholdFactor = (float*) ( m_oDistThresholdFrame.data + idx_flt32 );
      float* pfCurrVariationFactor = (float*) ( m_oVariationModulatorFrame.data + idx_flt32 );
      float* pfCurrLearningRate = ( (float*) ( m_oUpdateRateFrame.data + idx_flt32 ) );
      float* pfCurrMeanLastDist = ( (float*) ( m_oMeanLastDistFrame.data + idx_flt32 ) );
      float* pfCurrMeanMinDist_LT = ( (float*) ( m_oMeanMinDistFrame_LT.data + idx_flt32 ) );
      float* pfCurrMeanMinDist_ST = ( (float*) ( m_oMeanMinDistFrame_ST.data + idx_flt32 ) );
      float* pfCurrMeanRawSegmRes_LT = ( (float*) ( m_oMeanRawSegmResFrame_LT.data + idx_flt32 ) );
      float* pfCurrMeanRawSegmRes_ST = ( (float*) ( m_oMeanRawSegmResFrame_ST.data + idx_flt32 ) );
      float* pfCurrMeanFinalSegmRes_LT = ( (float*) ( m_oMeanFinalSegmResFrame_LT.data + idx_flt32 ) );
      float* pfCurrMeanFinalSegmRes_ST = ( (float*) ( m_oMeanFinalSegmResFrame_ST.data + idx_flt32 ) );
      ushort* anLastIntraDesc = ( (ushort*) ( m_oLastDescFrame.data + idx_ushrt_rgb ) );
      uchar* anLastColor = m_oLastColorFrame.data + idx_uchar_rgb;
      const size_t nCurrColorDistThreshold = (size_t) ( ( ( *pfCurrDistThresholdFactor ) * m_nMinColorDistThreshold )
          - ( ( !m_oUnstableRegionMask.data[idx_uchar] ) * STAB_COLOR_DIST_OFFSET ) );
      const size_t nCurrDescDistThreshold = ( (size_t) 1 << ( (size_t) floor( *pfCurrDistThresholdFactor + 0.5f ) ) ) + m_nDescDistThreshold
          + ( m_oUnstableRegionMask.data[idx_uchar] * UNSTAB_DESC_DIST_OFFSET );
      const size_t nCurrTotColorDistThreshold = nCurrColorDistThreshold * 3;
      const size_t nCurrTotDescDistThreshold = nCurrDescDistThreshold * 3;
      const size_t nCurrSCColorDistThreshold = nCurrTotColorDistThreshold / 2;
      ushort anCurrInterDesc[3], anCurrIntraDesc[3];
      const size_t anCurrIntraLBSPThresholds[3] =
      { m_anLBSPThreshold_8bitLUT[anCurrColor[0]], m_anLBSPThreshold_8bitLUT[anCurrColor[1]], m_anLBSPThreshold_8bitLUT[anCurrColor[2]] };
      LBSP::computeRGBDescriptor( oInputImg, anCurrColor, x, y, anCurrIntraLBSPThresholds, anCurrIntraDesc );
      m_oUnstableRegionMask.data[idx_uchar] =
          ( ( *pfCurrDistThresholdFactor ) > UNSTABLE_REG_RDIST_MIN
              || ( *pfCurrMeanRawSegmRes_LT - *pfCurrMeanFinalSegmRes_LT ) > UNSTABLE_REG_RATIO_MIN
              || ( *pfCurrMeanRawSegmRes_ST - *pfCurrMeanFinalSegmRes_ST ) > UNSTABLE_REG_RATIO_MIN ) ? 1 : 0;
      size_t nGoodSamplesCount = 0, nSampleIdx = 0;
      while ( nGoodSamplesCount < m_nRequiredBGSamples && nSampleIdx < m_nBGSamples )
      {
        const ushort* const anBGIntraDesc = (ushort*) ( m_voBGDescSamples[nSampleIdx].data + idx_ushrt_rgb );
        const uchar* const anBGColor = m_voBGColorSamples[nSampleIdx].data + idx_uchar_rgb;
        size_t nTotDescDist = 0;
        size_t nTotSumDist = 0;
        for ( size_t c = 0; c < 3; ++c )
        {
          const size_t nColorDist = absdiff_uchar( anCurrColor[c], anBGColor[c] );
          if( nColorDist > nCurrSCColorDistThreshold )
            goto failedcheck3ch;
          size_t nIntraDescDist = hdist_ushort_8bitLUT( anCurrIntraDesc[c], anBGIntraDesc[c] );
          LBSP::computeSingleRGBDescriptor( oInputImg, anBGColor[c], x, y, c, m_anLBSPThreshold_8bitLUT[anBGColor[c]], anCurrInterDesc[c] );
          size_t nInterDescDist = hdist_ushort_8bitLUT( anCurrInterDesc[c], anBGIntraDesc[c] );
          const size_t nDescDist = ( nIntraDescDist + nInterDescDist ) / 2;
          const size_t nSumDist = std::min( ( nDescDist / 2 ) * ( s_nColorMaxDataRange_1ch / s_nDescMaxDataRange_1ch ) + nColorDist,
                                            s_nColorMaxDataRange_1ch );
          if( nSumDist > nCurrSCColorDistThreshold )
            goto failedcheck3ch;
          nTotDescDist += nDescDist;
          nTotSumDist += nSumDist;
        }
        if( nTotDescDist > nCurrTotDescDistThreshold || nTotSumDist > nCurrTotColorDistThreshold )
          goto failedcheck3ch;
        if( nMinTotDescDist > nTotDescDist )
          nMinTotDescDist = nTotDescDist;
        if( nMinTotSumDist > nTotSumDist )
          nMinTotSumDist = nTotSumDist;
        nGoodSamplesCount++;
        failedcheck3ch : nSampleIdx++;
      }
      const float fNormalizedLastDist = ( (float) L1dist_uchar( anLastColor, anCurrColor ) / s_nColorMaxDataRange_3ch
          + (float) hdist_ushort_8bitLUT( anLastIntraDesc, anCurrIntraDesc ) / s_nDescMaxDataRange_3ch ) / 2;
      *pfCurrMeanLastDist = ( *pfCurrMeanLastDist ) * ( 1.0f - fRollAvgFactor_ST ) + fNormalizedLastDist * fRollAvgFactor_ST;
      if( nGoodSamplesCount < m_nRequiredBGSamples )
      {
        // == foreground
        const float fNormalizedMinDist = std::min(
            1.0f,
            ( (float) nMinTotSumDist / s_nColorMaxDataRange_3ch + (float) nMinTotDescDist / s_nDescMaxDataRange_3ch ) / 2
                + (float) ( m_nRequiredBGSamples - nGoodSamplesCount ) / m_nRequiredBGSamples );
        *pfCurrMeanMinDist_LT = ( *pfCurrMeanMinDist_LT ) * ( 1.0f - fRollAvgFactor_LT ) + fNormalizedMinDist * fRollAvgFactor_LT;
        *pfCurrMeanMinDist_ST = ( *pfCurrMeanMinDist_ST ) * ( 1.0f - fRollAvgFactor_ST ) + fNormalizedMinDist * fRollAvgFactor_ST;
        *pfCurrMeanRawSegmRes_LT = ( *pfCurrMeanRawSegmRes_LT ) * ( 1.0f - fRollAvgFactor_LT ) + fRollAvgFactor_LT;
        *pfCurrMeanRawSegmRes_ST = ( *pfCurrMeanRawSegmRes_ST ) * ( 1.0f - fRollAvgFactor_ST ) + fRollAvgFactor_ST;
        oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
        if( m_nModelResetCooldown && ( rand() % (size_t) FEEDBACK_T_LOWER ) == 0 )
        {
          const size_t s_rand = rand() % m_nBGSamples;
          for ( size_t c = 0; c < 3; ++c )
          {
            * ( (ushort*) ( m_voBGDescSamples[s_rand].data + idx_ushrt_rgb + 2 * c ) ) = anCurrIntraDesc[c];
            * ( m_voBGColorSamples[s_rand].data + idx_uchar_rgb + c ) = anCurrColor[c];
          }
        }
      }
      else
      {
        // == background
        const float fNormalizedMinDist = ( (float) nMinTotSumDist / s_nColorMaxDataRange_3ch + (float) nMinTotDescDist / s_nDescMaxDataRange_3ch )
            / 2;
        *pfCurrMeanMinDist_LT = ( *pfCurrMeanMinDist_LT ) * ( 1.0f - fRollAvgFactor_LT ) + fNormalizedMinDist * fRollAvgFactor_LT;
        *pfCurrMeanMinDist_ST = ( *pfCurrMeanMinDist_ST ) * ( 1.0f - fRollAvgFactor_ST ) + fNormalizedMinDist * fRollAvgFactor_ST;
        *pfCurrMeanRawSegmRes_LT = ( *pfCurrMeanRawSegmRes_LT ) * ( 1.0f - fRollAvgFactor_LT );
        *pfCurrMeanRawSegmRes_ST = ( *pfCurrMeanRawSegmRes_ST ) * ( 1.0f - fRollAvgFactor_ST );
        const size_t nLearningRate = learningRateOverride > 0 ? (size_t) ceil( learningRateOverride ) : (size_t) ceil( *pfCurrLearningRate );
        if( ( rand() % nLearningRate ) == 0 )
        {
          const size_t s_rand = rand() % m_nBGSamples;
          for ( size_t c = 0; c < 3; ++c )
          {
            * ( (ushort*) ( m_voBGDescSamples[s_rand].data + idx_ushrt_rgb + 2 * c ) ) = anCurrIntraDesc[c];
            * ( m_voBGColorSamples[s_rand].data + idx_uchar_rgb + c ) = anCurrColor[c];
          }
        }
        int x_rand, y_rand;
        const bool bCurrUsing3x3Spread = m_bUse3x3Spread && !m_oUnstableRegionMask.data[idx_uchar];
        if( bCurrUsing3x3Spread )
          getRandNeighborPosition_3x3( x_rand, y_rand, x, y, LBSP::PATCH_SIZE / 2, m_oImgSize );
        else
          getRandNeighborPosition_5x5( x_rand, y_rand, x, y, LBSP::PATCH_SIZE / 2, m_oImgSize );
        const size_t n_rand = rand();
        const size_t idx_rand_uchar = m_oImgSize.width * y_rand + x_rand;
        const size_t idx_rand_flt32 = idx_rand_uchar * 4;
        const float fRandMeanLastDist = * ( (float*) ( m_oMeanLastDistFrame.data + idx_rand_flt32 ) );
        const float fRandMeanRawSegmRes = * ( (float*) ( m_oMeanRawSegmResFrame_ST.data + idx_rand_flt32 ) );
        if( ( n_rand % ( bCurrUsing3x3Spread ? nLearningRate : ( nLearningRate / 2 + 1 ) ) ) == 0
            || ( fRandMeanRawSegmRes > GHOSTDET_S_MIN && fRandMeanLastDist < GHOSTDET_D_MAX
                && ( n_rand % ( (size_t) m_fCurrLearningRateLowerCap ) ) == 0 ) )
        {
          const size_t idx_rand_uchar_rgb = idx_rand_uchar * 3;
          const size_t idx_rand_ushrt_rgb = idx_rand_uchar_rgb * 2;
          const size_t s_rand = rand() % m_nBGSamples;
          for ( size_t c = 0; c < 3; ++c )
          {
            * ( (ushort*) ( m_voBGDescSamples[s_rand].data + idx_rand_ushrt_rgb + 2 * c ) ) = anCurrIntraDesc[c];
            * ( m_voBGColorSamples[s_rand].data + idx_rand_uchar_rgb + c ) = anCurrColor[c];
          }
        }
      }
      if( m_oFGMask_last.data[idx_uchar]
          || ( std::min( *pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST ) < UNSTABLE_REG_RATIO_MIN && oCurrFGMask.data[idx_uchar] ) )
      {
        if( ( *pfCurrLearningRate ) < m_fCurrLearningRateUpperCap )
          *pfCurrLearningRate += FEEDBACK_T_INCR / ( std::max( *pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST ) * ( *pfCurrVariationFactor ) );
      }
      else if( ( *pfCurrLearningRate ) > m_fCurrLearningRateLowerCap )
        *pfCurrLearningRate -= FEEDBACK_T_DECR * ( *pfCurrVariationFactor ) / std::max( *pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST );
      if( ( *pfCurrLearningRate ) < m_fCurrLearningRateLowerCap )
        *pfCurrLearningRate = m_fCurrLearningRateLowerCap;
      else if( ( *pfCurrLearningRate ) > m_fCurrLearningRateUpperCap )
        *pfCurrLearningRate = m_fCurrLearningRateUpperCap;
      if( std::max( *pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST ) > UNSTABLE_REG_RATIO_MIN && m_oBlinksFrame.data[idx_uchar] )
        ( *pfCurrVariationFactor ) += FEEDBACK_V_INCR;
      else if( ( *pfCurrVariationFactor ) > FEEDBACK_V_DECR )
      {
        ( *pfCurrVariationFactor ) -= m_oFGMask_last.data[idx_uchar] ? FEEDBACK_V_DECR / 4 :
                                      m_oUnstableRegionMask.data[idx_uchar] ? FEEDBACK_V_DECR / 2 : FEEDBACK_V_DECR;
        if( ( *pfCurrVariationFactor ) < FEEDBACK_V_DECR )
          ( *pfCurrVariationFactor ) = FEEDBACK_V_DECR;
      }
      if( ( *pfCurrDistThresholdFactor ) < std::pow( 1.0f + std::min( *pfCurrMeanMinDist_LT, *pfCurrMeanMinDist_ST ) * 2, 2 ) )
        ( *pfCurrDistThresholdFactor ) += FEEDBACK_R_VAR * ( *pfCurrVariationFactor - FEEDBACK_V_DECR );
      else
      {
        ( *pfCurrDistThresholdFactor ) -= FEEDBACK_R_VAR / ( *pfCurrVariationFactor );
        if( ( *pfCurrDistThresholdFactor ) < 1.0f )
          ( *pfCurrDistThresholdFactor ) = 1.0f;
      }
      if( popcount_ushort_8bitsLUT( anCurrIntraDesc ) >= 4 )
        ++nNonZeroDescCount;
      for ( size_t c = 0; c < 3; ++c )
      {
        anLastIntraDesc[c] = anCurrIntraDesc[c];
        anLastColor[c] = anCurrColor[c];
      }
    }
  }
#if DISPLAY_SUBSENSE_DEBUG_INFO
  std::cout << std::endl;
  Point dbgpt(nDebugCoordX,nDebugCoordY);
  Mat oMeanMinDistFrameNormalized; m_oMeanMinDistFrame_ST.copyTo(oMeanMinDistFrameNormalized);
  circle(oMeanMinDistFrameNormalized,dbgpt,5,Scalar(1.0f));
  resize(oMeanMinDistFrameNormalized,oMeanMinDistFrameNormalized,DEFAULT_FRAME_SIZE);
  imshow("d_min(x)",oMeanMinDistFrameNormalized);
  std::cout << std::fixed << std::setprecision(5) << "  d_min(" << dbgpt << ") = " << m_oMeanMinDistFrame_ST.at<float>(dbgpt) << std::endl;
  Mat oMeanLastDistFrameNormalized; m_oMeanLastDistFrame.copyTo(oMeanLastDistFrameNormalized);
  circle(oMeanLastDistFrameNormalized,dbgpt,5,Scalar(1.0f));
  resize(oMeanLastDistFrameNormalized,oMeanLastDistFrameNormalized,DEFAULT_FRAME_SIZE);
  imshow("d_last(x)",oMeanLastDistFrameNormalized);
  std::cout << std::fixed << std::setprecision(5) << " d_last(" << dbgpt << ") = " << m_oMeanLastDistFrame.at<float>(dbgpt) << std::endl;
  Mat oMeanRawSegmResFrameNormalized; m_oMeanRawSegmResFrame_ST.copyTo(oMeanRawSegmResFrameNormalized);
  circle(oMeanRawSegmResFrameNormalized,dbgpt,5,Scalar(1.0f));
  resize(oMeanRawSegmResFrameNormalized,oMeanRawSegmResFrameNormalized,DEFAULT_FRAME_SIZE);
  imshow("s_avg(x)",oMeanRawSegmResFrameNormalized);
  std::cout << std::fixed << std::setprecision(5) << "  s_avg(" << dbgpt << ") = " << m_oMeanRawSegmResFrame_ST.at<float>(dbgpt) << std::endl;
  Mat oMeanFinalSegmResFrameNormalized; m_oMeanFinalSegmResFrame_ST.copyTo(oMeanFinalSegmResFrameNormalized);
  circle(oMeanFinalSegmResFrameNormalized,dbgpt,5,Scalar(1.0f));
  resize(oMeanFinalSegmResFrameNormalized,oMeanFinalSegmResFrameNormalized,DEFAULT_FRAME_SIZE);
  imshow("z_avg(x)",oMeanFinalSegmResFrameNormalized);
  std::cout << std::fixed << std::setprecision(5) << "  z_avg(" << dbgpt << ") = " << m_oMeanFinalSegmResFrame_ST.at<float>(dbgpt) << std::endl;
  Mat oDistThresholdFrameNormalized; m_oDistThresholdFrame.convertTo(oDistThresholdFrameNormalized,CV_32FC1,0.25f,-0.25f);
  circle(oDistThresholdFrameNormalized,dbgpt,5,Scalar(1.0f));
  resize(oDistThresholdFrameNormalized,oDistThresholdFrameNormalized,DEFAULT_FRAME_SIZE);
  imshow("r(x)",oDistThresholdFrameNormalized);
  std::cout << std::fixed << std::setprecision(5) << "      r(" << dbgpt << ") = " << m_oDistThresholdFrame.at<float>(dbgpt) << std::endl;
  Mat oVariationModulatorFrameNormalized; normalize(m_oVariationModulatorFrame,oVariationModulatorFrameNormalized,0,255,NORM_MINMAX,CV_8UC1);
  circle(oVariationModulatorFrameNormalized,dbgpt,5,Scalar(255));
  resize(oVariationModulatorFrameNormalized,oVariationModulatorFrameNormalized,DEFAULT_FRAME_SIZE);
  imshow("v(x)",oVariationModulatorFrameNormalized);
  std::cout << std::fixed << std::setprecision(5) << "      v(" << dbgpt << ") = " << m_oVariationModulatorFrame.at<float>(dbgpt) << std::endl;
  Mat oUpdateRateFrameNormalized; m_oUpdateRateFrame.convertTo(oUpdateRateFrameNormalized,CV_32FC1,1.0f/FEEDBACK_T_UPPER,-FEEDBACK_T_LOWER/FEEDBACK_T_UPPER);
  circle(oUpdateRateFrameNormalized,dbgpt,5,Scalar(1.0f));
  resize(oUpdateRateFrameNormalized,oUpdateRateFrameNormalized,DEFAULT_FRAME_SIZE);
  imshow("t(x)",oUpdateRateFrameNormalized);
  std::cout << std::fixed << std::setprecision(5) << "      t(" << dbgpt << ") = " << m_oUpdateRateFrame.at<float>(dbgpt) << std::endl;
#endif //DISPLAY_SUBSENSE_DEBUG_INFO
  bitwise_xor( oCurrFGMask, m_oRawFGMask_last, m_oRawFGBlinkMask_curr );
  bitwise_or( m_oRawFGBlinkMask_curr, m_oRawFGBlinkMask_last, m_oBlinksFrame );
  m_oRawFGBlinkMask_curr.copyTo( m_oRawFGBlinkMask_last );
  oCurrFGMask.copyTo( m_oRawFGMask_last );
  morphologyEx( oCurrFGMask, m_oFGMask_PreFlood, MORPH_CLOSE, Mat() );
  m_oFGMask_PreFlood.copyTo( m_oFGMask_FloodedHoles );
  floodFill( m_oFGMask_FloodedHoles, Point( 0, 0 ), UCHAR_MAX );
  bitwise_not( m_oFGMask_FloodedHoles, m_oFGMask_FloodedHoles );
  erode( m_oFGMask_PreFlood, m_oFGMask_PreFlood, Mat(), Point( -1, -1 ), 3 );
  bitwise_or( oCurrFGMask, m_oFGMask_FloodedHoles, oCurrFGMask );
  bitwise_or( oCurrFGMask, m_oFGMask_PreFlood, oCurrFGMask );
  medianBlur( oCurrFGMask, m_oFGMask_last, m_nMedianBlurKernelSize );
  dilate( m_oFGMask_last, m_oFGMask_last_dilated, Mat(), Point( -1, -1 ), 3 );
  bitwise_and( m_oBlinksFrame, m_oFGMask_last_dilated_inverted, m_oBlinksFrame );
  bitwise_not( m_oFGMask_last_dilated, m_oFGMask_last_dilated_inverted );
  bitwise_and( m_oBlinksFrame, m_oFGMask_last_dilated_inverted, m_oBlinksFrame );
  m_oFGMask_last.copyTo( oCurrFGMask );
  addWeighted( m_oMeanFinalSegmResFrame_LT, ( 1.0f - fRollAvgFactor_LT ), m_oFGMask_last, ( 1.0 / UCHAR_MAX ) * fRollAvgFactor_LT, 0,
               m_oMeanFinalSegmResFrame_LT, CV_32F );
  addWeighted( m_oMeanFinalSegmResFrame_ST, ( 1.0f - fRollAvgFactor_ST ), m_oFGMask_last, ( 1.0 / UCHAR_MAX ) * fRollAvgFactor_ST, 0,
               m_oMeanFinalSegmResFrame_ST, CV_32F );
  const float fCurrNonZeroDescRatio = (float) nNonZeroDescCount / m_nKeyPoints;
  if( fCurrNonZeroDescRatio < LBSPDESC_NONZERO_RATIO_MIN && m_fLastNonZeroDescRatio < LBSPDESC_NONZERO_RATIO_MIN )
  {
    for ( size_t t = 0; t <= UCHAR_MAX; ++t )
      if( m_anLBSPThreshold_8bitLUT[t] > saturate_cast<uchar>( m_nLBSPThresholdOffset + ceil( t * m_fRelLBSPThreshold / 4 ) ) )
        --m_anLBSPThreshold_8bitLUT[t];
  }
  else if( fCurrNonZeroDescRatio > LBSPDESC_NONZERO_RATIO_MAX && m_fLastNonZeroDescRatio > LBSPDESC_NONZERO_RATIO_MAX )
  {
    for ( size_t t = 0; t <= UCHAR_MAX; ++t )
      if( m_anLBSPThreshold_8bitLUT[t] < saturate_cast<uchar>( m_nLBSPThresholdOffset + UCHAR_MAX * m_fRelLBSPThreshold ) )
        ++m_anLBSPThreshold_8bitLUT[t];
  }
  m_fLastNonZeroDescRatio = fCurrNonZeroDescRatio;
  if( m_bLearningRateScalingEnabled )
  {
    resize( oInputImg, m_oDownSampledColorFrame, m_oDownSampledFrameSize, 0, 0, INTER_AREA );
    accumulateWeighted( m_oDownSampledColorFrame, m_oMeanDownSampledLastDistFrame_LT, fRollAvgFactor_LT );
    accumulateWeighted( m_oDownSampledColorFrame, m_oMeanDownSampledLastDistFrame_ST, fRollAvgFactor_ST );
    size_t nTotColorDiff = 0;
    for ( int i = 0; i < m_oMeanDownSampledLastDistFrame_ST.rows; ++i )
    {
      const size_t idx1 = m_oMeanDownSampledLastDistFrame_ST.step.p[0] * i;
      for ( int j = 0; j < m_oMeanDownSampledLastDistFrame_ST.cols; ++j )
      {
        const size_t idx2 = idx1 + m_oMeanDownSampledLastDistFrame_ST.step.p[1] * j;
        nTotColorDiff +=
            ( m_nImgChannels == 1 ) ?
                (size_t) fabs(
                    ( *(float*) ( m_oMeanDownSampledLastDistFrame_ST.data + idx2 ) )
                        - ( *(float*) ( m_oMeanDownSampledLastDistFrame_LT.data + idx2 ) ) ) / 2 :  //(m_nImgChannels==3)
                std::max(
                    (size_t) fabs(
                        ( *(float*) ( m_oMeanDownSampledLastDistFrame_ST.data + idx2 ) )
                            - ( *(float*) ( m_oMeanDownSampledLastDistFrame_LT.data + idx2 ) ) ),
                    std::max(
                        (size_t) fabs(
                            ( *(float*) ( m_oMeanDownSampledLastDistFrame_ST.data + idx2 + 4 ) )
                                - ( *(float*) ( m_oMeanDownSampledLastDistFrame_LT.data + idx2 + 4 ) ) ),
                        (size_t) fabs(
                            ( *(float*) ( m_oMeanDownSampledLastDistFrame_ST.data + idx2 + 8 ) )
                                - ( *(float*) ( m_oMeanDownSampledLastDistFrame_LT.data + idx2 + 8 ) ) ) ) );
      }
    }
    const float fCurrColorDiffRatio = (float) nTotColorDiff / ( m_oMeanDownSampledLastDistFrame_ST.rows * m_oMeanDownSampledLastDistFrame_ST.cols );
    if( m_bAutoModelResetEnabled )
    {
      if( m_nFramesSinceLastReset > 1000 )
        m_bAutoModelResetEnabled = false;
      else if( fCurrColorDiffRatio >= FRAMELEVEL_COLOR_DIFF_RESET_THRESHOLD && m_nModelResetCooldown == 0 )
      {
        m_nFramesSinceLastReset = 0;
        refreshModel( 0.1f );  // reset 10% of the bg model
        m_nModelResetCooldown = m_nSamplesForMovingAvgs;
        m_oUpdateRateFrame = Scalar( 1.0f );
      }
      else
        ++m_nFramesSinceLastReset;
    }
    else if( fCurrColorDiffRatio >= FRAMELEVEL_COLOR_DIFF_RESET_THRESHOLD * 2 )
    {
      m_nFramesSinceLastReset = 0;
      m_bAutoModelResetEnabled = true;
    }
    if( fCurrColorDiffRatio >= FRAMELEVEL_COLOR_DIFF_RESET_THRESHOLD / 2 )
    {
      m_fCurrLearningRateLowerCap = (float) std::max( (int) FEEDBACK_T_LOWER >> (int) ( fCurrColorDiffRatio / 2 ), 1 );
      m_fCurrLearningRateUpperCap = (float) std::max( (int) FEEDBACK_T_UPPER >> (int) ( fCurrColorDiffRatio / 2 ), 1 );
    }
    else
    {
      m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER;
      m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER;
    }
    if( m_nModelResetCooldown > 0 )
      --m_nModelResetCooldown;
  }
}

void MotionSaliencySuBSENSE::getBackgroundImage( OutputArray backgroundImage ) const
{
  CV_Assert( m_bInitialized );
  Mat oAvgBGImg = Mat::zeros( m_oImgSize, CV_32FC( (int )m_nImgChannels ) );
  for ( size_t s = 0; s < m_nBGSamples; ++s )
  {
    for ( int y = 0; y < m_oImgSize.height; ++y )
    {
      for ( int x = 0; x < m_oImgSize.width; ++x )
      {
        const size_t idx_nimg = m_voBGColorSamples[s].step.p[0] * y + m_voBGColorSamples[s].step.p[1] * x;
        const size_t idx_flt32 = idx_nimg * 4;
        float* oAvgBgImgPtr = (float*) ( oAvgBGImg.data + idx_flt32 );
        const uchar* const oBGImgPtr = m_voBGColorSamples[s].data + idx_nimg;
        for ( size_t c = 0; c < m_nImgChannels; ++c )
          oAvgBgImgPtr[c] += ( (float) oBGImgPtr[c] ) / m_nBGSamples;
      }
    }
  }
  oAvgBGImg.convertTo( backgroundImage, CV_8U );
}

void MotionSaliencySuBSENSE::setAutomaticModelReset( bool b )
{
  m_bAutoModelResetEnabled = b;
}

}/* namespace cv */
