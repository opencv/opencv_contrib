// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

Shift::Shift(float _amount)
{
  CV_Assert(_amount >= 0 && _amount <= 1);
  amount = { _amount, _amount };
  randomAmount = false;
}

Shift::Shift(Vec2f _amount)
{
  CV_Assert(_amount[0] >= 0 && _amount[0] <= 1 && _amount[1] >= 0 && _amount[1] <= 1);
  amount = _amount;
  randomAmount = false;
}

Shift::Shift(Vec2f _minAmount, Vec2f _maxAmount)
{
  CV_Assert(_minAmount[0] >= 0 && _minAmount[0] <= 1 && _minAmount[1] >= 0 && _minAmount[1] <= 1);
  CV_Assert(_maxAmount[0] >= 0 && _maxAmount[0] <= 1 && _maxAmount[1] >= 0 && _maxAmount[1] <= 1);
  minAmount = _minAmount;
  maxAmount = _maxAmount;
  randomAmount = true;
}

Shift::Shift(float _minAmount, float _maxAmount)
{
  CV_Assert(_minAmount >= 0 && _minAmount <= 1 && _maxAmount >= 0 && _maxAmount <= 1);
  minAmount = { _minAmount, _minAmount };
  maxAmount = {_maxAmount, _maxAmount};
  randomAmount = true;
}

void Shift::init(const Mat& srcImage)
{
  Transform::init(srcImage);
  if (randomAmount)
  {
    columnsShifted = int(Transform::rng.uniform(minAmount[0], maxAmount[0]) * srcImageCols);
    rowsShifted = int(Transform::rng.uniform(minAmount[1], maxAmount[1]) * srcImageRows);
  }

  else
  {
    columnsShifted = int(amount[0] * srcImageCols);
    rowsShifted = int(amount[1] * srcImageRows);
  }

  translationMat = (Mat_<double>(2, 3) << 1, 0, columnsShifted, 0, 1, rowsShifted);
}

void Shift::image(InputArray src, OutputArray dst)
{
  warpAffine(src, dst, translationMat, src.size());
}

Point2f Shift::point(const Point2f& src)
{
  return Point2f(src.x + columnsShifted, src.y + rowsShifted);
}

Rect2f Shift::rectangle(const Rect2f& src)
{
  return Rect2f(src.x + columnsShifted, src.y + rowsShifted, src.width, src.height);
}
}}
