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

//TODO delete
#define SALIENCY_DEBUG true
#ifdef SALIENCY_DEBUG
#include <opencv2/highgui.hpp>
#endif

namespace cv
{

/************************************ Specific Static Saliency Specialized Classes ************************************/

/**
 * \brief Saliency based on algorithms based on [1]
 * [1] Yan, Jia, Xi Chen, and QiuPing Zhu. "Robust online tracking via adaptive samples selection with saliency detection." (2013)
 */
class CV_EXPORTS_W StaticSaliencySpectralResidual : public StaticSaliency
{
 public:
  struct CV_EXPORTS Params
  {
    Params();
    void read( const FileNode& fn );
    void write( FileStorage& fs ) const;
  };
  StaticSaliencySpectralResidual( const StaticSaliencySpectralResidual::Params &parameters = StaticSaliencySpectralResidual::Params() );
  ~StaticSaliencySpectralResidual();

  void read( const FileNode& fn );
  void write( FileStorage& fs ) const;

 protected:
  bool computeKmeans( Mat& saliencyMap, Mat& outputMat );
  bool computeSaliencyImpl( const Mat& src, Mat& dst );
  AlgorithmInfo* info() const;

 private:
  Params params;
};

/************************************ Specific Motion Saliency Specialized Classes ************************************/

/**
 * \brief Saliency based on algorithms based on [2]
 * [2] Hofmann, Martin, Philipp Tiefenbacher, and Gerhard Rigoll. "Background segmentation with feedback: The pixel-based adaptive segmenter."
 *     Computer Vision and Pattern Recognition Workshops (CVPRW), 2012 IEEE Computer Society Conference on. IEEE, 2012.
 */
class CV_EXPORTS_W MotionSaliencyPBAS : public MotionSaliency
{
 public:
  struct CV_EXPORTS Params
  {
    Params();
    void read( const FileNode& fn );
    void write( FileStorage& fs ) const;
  };
  //MotionSaliencyPBAS(const MotionSaliencyPBAS::Params &parameters = MotionSaliencyPBAS::Params());
  MotionSaliencyPBAS();
  ~MotionSaliencyPBAS();

  void read( const FileNode& fn );
  void write( FileStorage& fs ) const;

 protected:
  bool computeSaliencyImpl( const Mat& src, Mat& dst );
  AlgorithmInfo* info() const;

 private:
  Params params;
};

/************************************ Specific Objectness Specialized Classes ************************************/

/**
 * \brief Objectness algorithms based on [3]
 * [3] Cheng, Ming-Ming, et al. "BING: Binarized normed gradients for objectness estimation at 300fps." IEEE CVPR. 2014.
 */
class CV_EXPORTS_W ObjectnessBING : public Objectness
{
 public:
  struct CV_EXPORTS Params
  {
    Params();
    void read( const FileNode& fn );
    void write( FileStorage& fs ) const;
  };
  //ObjectnessBING(const ObjectnessBING::Params &parameters = ObjectnessBING::Params());
  ObjectnessBING();
  ~ObjectnessBING();

  void read( const FileNode& fn );
  void write( FileStorage& fs ) const;

 protected:
  bool computeSaliencyImpl( const Mat& src, Mat& dst );
  AlgorithmInfo* info() const;

 private:
  Params params;
};

} /* namespace cv */

#endif
