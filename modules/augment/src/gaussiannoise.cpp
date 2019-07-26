// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
namespace cv { namespace augment {

GaussianNoise::GaussianNoise(float _mean, float _stddev)
{
  mean = _mean;
  stddev = _stddev;
}

void GaussianNoise::image(InputArray src, OutputArray dst)
{
  addWeighted(src, 1.0, noise, 1.0, 0.0, dst);
}

Rect2f GaussianNoise::rectangle(const Rect2f& src)
{
  return src;
}

void GaussianNoise::init(const Mat& srcImage)
{
  noise = Mat(srcImage.size(), srcImage.type(),Scalar(0));
  randn(noise, mean, stddev);
}

}}
