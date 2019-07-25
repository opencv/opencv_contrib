// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

UniformNoise::UniformNoise(float _low, float _high)
{
  low = _low;
  high = _high;
}

void UniformNoise::image(InputArray src, OutputArray dst)
{
  addWeighted(src, 1.0, noise, 1.0, 0.0, dst);
}

Rect2f UniformNoise::rectangle(const Rect2f& src)
{
  return src;
}

void UniformNoise::init(const Mat& srcImage)
{
  noise = Mat(srcImage.size(), srcImage.type(),Scalar(0));
  randu(noise, low, high);
}

}}
