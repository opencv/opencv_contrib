// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv::linemod;

// Regression test for opencv/opencv#29559: linemod::Detector::match() crashed
// with SIGSEGV inside orUnaligned8u() (called from the static spread(), only
// reachable from match()'s pyramid-building step) whenever a pyramid level's
// quantized image had a row width that was not a multiple of 16.
// orUnaligned8u() used to compute src/dst SSE alignment once per call and
// reuse that stale decision for every row, taking an aligned SSE load/store
// path even after the row stride had drifted the pointer out of alignment.
// The fix removes that unsound alignment fast path.
//
// spread()/orUnaligned8u() are private (static) implementation details, not
// part of the public API, so this test drives them indirectly through
// Detector::match() -- which unconditionally rebuilds its linear-memory
// pyramid (and therefore calls spread()) for every match() call, regardless
// of whether any template was added -- reproducing the original crash
// through the same call path real callers use, per opencv/opencv#29559.
TEST(Rgbd_Linemod, match_survives_non16aligned_row_width)
{
  // Width intentionally not a multiple of 16 at both pyramid levels (level 0
  // is the raw image; level 1 is pyrDown()'d to roughly half size), so each
  // level's quantized image row stride is also not 16-byte aligned --
  // matching the actual production trigger, not merely an initially-offset
  // pointer. 34 -> pyrDown -> 17; neither is a multiple of 16.
  const int width = 34;
  const int height = 64; // -> pyrDown -> 32

  // computeResponseMaps() asserts (rows*cols) % 16 == 0 at every pyramid
  // level; this is unrelated to per-row stride alignment (the actual bug
  // condition) but must still hold for the test image to be valid input.
  // Level 0: 64*34 = 2176 = 136*16. Level 1: 32*17 = 544 = 34*16.
  ASSERT_EQ((height * width) % 16, 0);
  ASSERT_EQ((((height + 1) / 2) * ((width + 1) / 2)) % 16, 0);

  Mat src(height, width, CV_8UC3, Scalar(0, 0, 0));
  // High-contrast checkerboard so ColorGradient finds strong, plentiful
  // gradients on this exact (non-16-aligned) image size, independent of any
  // specific pixel statistics.
  const int block = 4;
  for (int r = 0; r < height; ++r)
    for (int c = 0; c < width; ++c)
      if (((r / block) + (c / block)) % 2 == 0)
        src.at<Vec3b>(r, c) = Vec3b(255, 255, 255);

  Mat mask(height, width, CV_8U, Scalar(255));

  // Lenient thresholds so template extraction succeeds reliably on the
  // synthetic checkerboard regardless of its exact statistics; this test is
  // about alignment safety, not feature-extraction tuning.
  std::vector< Ptr<Modality> > modalities;
  modalities.push_back(ColorGradient::create(1.0f, 4, 1.0f));
  int T_vals[] = {5, 8}; // same T_pyramid as getDefaultLINE()
  Detector detector(modalities, std::vector<int>(T_vals, T_vals + 2));

  std::vector<Mat> sources(1, src);

  int template_id = detector.addTemplate(sources, "checkerboard", mask);
  ASSERT_GE(template_id, 0) << "template extraction failed on the synthetic checkerboard image";

  std::vector<Match> matches;
  // The call that reaches spread()/orUnaligned8u on every pyramid level;
  // pre-fix, this crashed with SIGSEGV whenever a level's quantized image
  // row width wasn't 16-byte aligned.
  detector.match(sources, /*threshold=*/50.f, matches);

  ASSERT_FALSE(matches.empty())
      << "expected the detector to match its own just-added template against the same image";
  EXPECT_GT(matches[0].similarity, 90.f);
  EXPECT_EQ(matches[0].class_id, "checkerboard");
}

}} // namespace
