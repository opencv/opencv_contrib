#pragma once

#ifndef _OPENCV_PHOTOEFFECTS_HPP_
#define _OPENCV_PHOTOEFFECTS_HPP_
#ifdef __cplusplus

#include <opencv2/core/core.hpp>

namespace cv { namespace photoeffects {

int sepia(cv::InputArray src, cv::OutputArray dst);

int filmGrain(cv::InputArray src, cv::OutputArray dst, int grainValue = 8, cv::RNG& rng = cv::theRNG());

int fadeColor(cv::InputArray src, cv::OutputArray dst,cv::Point startPoint,cv::Point endPoint);

int tint(cv::InputArray src, cv::OutputArray dst, const cv::Vec3b &colorTint, float density);

int glow(cv::InputArray src, cv::OutputArray dst, int radius = 0, float intensity = 0.0f);

int edgeBlur(cv::InputArray src, cv::OutputArray dst, int indentTop, int indentLeft);

int boostColor(cv::InputArray src, cv::OutputArray dst, float intensity = 0.0f);

int antique(cv::InputArray src, cv::OutputArray dst, cv::InputArray texture, float alpha);

int vignette(cv::InputArray src, cv::OutputArray dst, cv::Size rect);

int warmify(cv::InputArray src, cv::OutputArray dst, uchar delta = 30);

int matte(cv::InputArray src, cv::OutputArray dst, cv::Point firstPoint, cv::Point secondPoint,
          float sigmaX, float sigmaY, cv::Size ksize=cv::Size(0, 0));

}}

#endif
#endif
