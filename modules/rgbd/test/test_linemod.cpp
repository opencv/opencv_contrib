// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

// orUnaligned8u() and spread() are private (static) implementation details of
// linemod.cpp, not part of the public cv::linemod API. Regression coverage for
// opencv/opencv#29559 needs to drive spread()'s exact internal call pattern --
// specifically a destination row stride that is not a multiple of 16, which is
// what makes orUnaligned8u's dst/src pointers drift out of 16-byte alignment
// across rows -- so this test includes the source translation unit directly to
// reach spread(), mirroring how some OpenCV modules test static internals.
#include "../src/linemod.cpp"

namespace opencv_test { namespace {

using namespace cv::linemod;

// Regression test for opencv/opencv#29559: linemod::match() crashed with
// SIGSEGV inside orUnaligned8u() (called from spread()) whenever the
// quantized image's row width was not a multiple of 16. orUnaligned8u used
// to compute src/dst alignment once per call and reuse that stale decision
// for every row, taking an aligned SSE load/store path even after the row
// stride had drifted the pointer out of alignment. The fix removes that
// unsound alignment fast path; this test exercises spread() with a row
// width that is not 16-byte aligned and would have crashed pre-fix.
TEST(Rgbd_Linemod, spread_handles_non16aligned_row_stride)
{
  // Width intentionally not a multiple of 16 so that dst.step1() (the
  // destination row stride orUnaligned8u advances by) is also not a
  // multiple of 16, causing per-row alignment drift -- the actual
  // production trigger, not merely an initially-offset pointer.
  const int width = 17;
  const int height = 20;
  const int T = 4; // spread() loops r,c in [0, T), each iteration calls
                   // orUnaligned8u across `height - r` rows -- enough rows
                   // for the drifted alignment to be exercised repeatedly.

  Mat quantized(height, width, CV_8U);
  cv::RNG rng(12345);
  // Populate with a mix of single-bit-set quantized orientation labels
  // (valid inputs to spread()) so the OR accumulation is non-trivial.
  static const uchar labels[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  for (int r = 0; r < height; ++r)
    for (int c = 0; c < width; ++c)
      quantized.at<uchar>(r, c) = labels[rng.uniform(0, 8)];

  Mat spread_quantized;
  ASSERT_NO_THROW(spread(quantized, spread_quantized, T));

  ASSERT_EQ(spread_quantized.size(), quantized.size());
  ASSERT_EQ(spread_quantized.type(), CV_8U);

  // Cross-check against an independent scalar reimplementation of spread()'s
  // documented behavior (OR quantized labels over a T x T neighborhood
  // shifted by each (r, c) offset) to confirm the SIMD path's fixed
  // alignment handling still produces numerically correct output, not just
  // a non-crashing one.
  Mat expected = Mat::zeros(quantized.size(), CV_8U);
  for (int dr = 0; dr < T; ++dr)
  {
    int rows = height - dr;
    for (int dc = 0; dc < T; ++dc)
    {
      int cols = width - dc;
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
          expected.at<uchar>(r, c) |= quantized.at<uchar>(r + dr, c + dc);
    }
  }

  ASSERT_EQ(cv::countNonZero(expected != spread_quantized), 0)
      << "spread() output mismatch on a non-16-aligned row width -- "
         "orUnaligned8u's fixed alignment handling must produce the same "
         "result as before, not merely avoid crashing.";
}

}} // namespace
