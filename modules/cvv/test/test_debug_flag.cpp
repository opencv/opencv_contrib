// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

/**
 * Tests whether cvv::debugMode() and cvv::setDebugFlag(bool)`  
 * (from /include/opencv2/debug_mode.hpp) behave correctly.
 */

TEST(DebugFlagTest, SetAndUnsetDebugMode)
{
	EXPECT_EQ(cvv::debugMode(), true);
	cvv::setDebugFlag(false);
	EXPECT_EQ(cvv::debugMode(), false);
	cvv::setDebugFlag(true);
	EXPECT_EQ(cvv::debugMode(), true);
}

}} // namespace
