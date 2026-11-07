// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv::linemod;

// Regression test for opencv/opencv#29559: linemod::Detector::match() crashed
// with SIGSEGV inside orUnaligned8u() (called from the static spread(), only
// reachable from match()'s pyramid-building step) whenever a pyramid level's
// quantized image had a row width that was not a multiple of 16. Only one
// pyramid level needs a non-16-aligned width to reach the buggy code path
// (see the width/height comment below for why both levels can't have it
// simultaneously under this test's T_pyramid).
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
  // Detector::match() also requires, via linearize(), that each pyramid
  // level's quantized image size be an exact multiple of that level's T
  // (T_pyramid = {5, 8} below, matching getDefaultLINE()). Since pyrDown()
  // here halves dimensions exactly, requiring level 1's width to be a
  // multiple of 8 forces level 0's width to be a multiple of 16 -- so level
  // 0 and level 1 can never *both* be non-16-aligned under this T_pyramid.
  // Only level 1's width is made non-16-aligned here (that alone is enough
  // to reach the buggy code path, per opencv/opencv#29559); level 0 comes
  // out 16-aligned as an unavoidable side effect of the T=8 constraint, not
  // by choice.
  // The image also needs to be large relative to matchClass()'s fixed
  // `8 * T` search-refinement border (T=5 at level 0): too small an image
  // leaves no room for the coarse-to-fine match search to converge, causing
  // spurious near-zero-similarity results unrelated to alignment at all.
  const int width = 240;   // -> pyrDown -> 120
  const int height = 240;  // -> pyrDown -> 120

  // computeResponseMaps() asserts (rows*cols) % 16 == 0 at every pyramid
  // level; this is unrelated to per-row stride alignment (the actual bug
  // condition) but must still hold for the test image to be valid input.
  // Level 0: 240*240 = 57600 = 3600*16. Level 1: 120*120 = 14400 = 900*16.
  ASSERT_EQ((height * width) % 16, 0);
  ASSERT_EQ(((height / 2) * (width / 2)) % 16, 0);
  // linearize()'s T-alignment requirement (see comment above).
  ASSERT_EQ(width % 5, 0) << "test premise requires level 0 dimensions divisible by T=5";
  ASSERT_EQ(height % 5, 0) << "test premise requires level 0 dimensions divisible by T=5";
  ASSERT_EQ((width / 2) % 8, 0) << "test premise requires level 1 dimensions divisible by T=8";
  ASSERT_EQ((height / 2) % 8, 0) << "test premise requires level 1 dimensions divisible by T=8";
  // Guard the test's own premise: if cv::Mat's row-stride convention ever
  // changed to pad rows to a 16-byte boundary, this test would silently stop
  // exercising the bug it exists to catch while still passing. Fail loudly
  // instead.
  ASSERT_NE((width / 2) % 16, 0) << "test premise requires a non-16-aligned row width at level 1";

  // On non-x86 hardware (e.g. arm64), orUnaligned8u() never compiles the SSE
  // load/store paths this PR fixes -- CV_SSE2/CV_SSE3 are false, so only its
  // plain scalar tail loop runs. A green run there exercises match()'s
  // control flow but proves nothing about the SIMD alignment-drift bug this
  // test exists to catch. Skip explicitly (rather than silently passing) so
  // CI output cannot be mistaken for SIMD coverage on such runners; on
  // SSE2-capable hardware this is a real, unconditional gate.
  if (!cv::checkHardwareSupport(CV_CPU_SSE2))
    throw cvtest::SkipTestException("orUnaligned8u()'s SSE load/store paths are not compiled/available "
                                     "on this CPU (no SSE2); this regression test cannot exercise the "
                                     "alignment-drift bug it targets here.");

  Mat src(height, width, CV_8UC3, Scalar(0, 0, 0));
  // High-contrast checkerboard confined to a centered sub-region, surrounded
  // by a solid black margin, rather than filling the whole frame. Two
  // reasons: (1) matchClass()'s pyramid-refinement step requires free search
  // space of `8 * T` pixels around the object at the finest level (T=5
  // here), so an edge-to-edge object leaves no room to relocate a candidate
  // match and the search silently fails; (2) with no mask (see below),
  // extractTemplate() finds candidate features wherever gradients exist --
  // confining the checkerboard keeps candidates (and therefore the
  // template's bounding box) tightly inside the margin, matching the
  // border requirement above by construction.
  const int block = 4;
  const int object_size = 96; // multiple of `block`; leaves >= 72px margin on every side
  const int object_off_r = (height - object_size) / 2;
  const int object_off_c = (width - object_size) / 2;
  for (int r = 0; r < object_size; ++r)
    for (int c = 0; c < object_size; ++c)
      if (((r / block) + (c / block)) % 2 == 0)
        src.at<Vec3b>(object_off_r + r, object_off_c + c) = Vec3b(255, 255, 255);

  // No mask: every pixel is a feature candidate, but the solid black margin
  // around the checkerboard has zero gradient everywhere, so candidates (and
  // the resulting template bounding box) are naturally confined to the
  // checkerboard region -- no mask-border-ring interaction to reason about.
  Mat mask;

  // Lenient thresholds so template extraction succeeds reliably on the
  // synthetic checkerboard regardless of its exact statistics; this test is
  // about alignment safety, not feature-extraction tuning.
  std::vector< Ptr<Modality> > modalities;
  modalities.push_back(ColorGradient::create(1.0f, 20, 1.0f));
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
