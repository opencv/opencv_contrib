#pragma once

#ifndef _OPENCV_PHOTOEFFECTS_HPP_
#define _OPENCV_PHOTOEFFECTS_HPP_
#ifdef __cplusplus

#include <opencv2/core.hpp>

namespace cv { namespace photoeffects {

CV_EXPORTS_W  int sepia(cv::InputArray src, cv::OutputArray dst);

CV_EXPORTS_W  void filmGrain(cv::InputArray src, cv::OutputArray dst, int grainValue = 8, cv::RNG& rng = cv::theRNG());

CV_EXPORTS_W  void fadeColor(cv::InputArray src, cv::OutputArray dst,cv::Point startPoint,cv::Point endPoint);

CV_EXPORTS_W  void tint(cv::InputArray src, cv::OutputArray dst, const cv::Vec3b &colorTint, float density);

CV_EXPORTS_W  void glow(cv::InputArray src, cv::OutputArray dst, int radius = 0, float intensity = 0.0f);

CV_EXPORTS_W  void edgeBlur(cv::InputArray src, cv::OutputArray dst, int indentTop, int indentLeft);

CV_EXPORTS_W  void boostColor(cv::InputArray src, cv::OutputArray dst, float intensity = 0.0f);

CV_EXPORTS_W  int antique(cv::InputArray src, cv::OutputArray dst, cv::InputArray texture, float alpha);

CV_EXPORTS_W  void vignette(cv::InputArray src, cv::OutputArray dst, cv::Size rect);

CV_EXPORTS_W  int warmify(cv::InputArray src, cv::OutputArray dst, uchar delta = 30);

CV_EXPORTS_W  int matte(cv::InputArray src, cv::OutputArray dst, cv::Point firstPoint, cv::Point secondPoint,
          float sigmaX, float sigmaY, cv::Size ksize=cv::Size(0, 0));

}}

#endif
#endif
