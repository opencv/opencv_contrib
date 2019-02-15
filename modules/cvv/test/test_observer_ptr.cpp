// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

#include "../src/util/observer_ptr.hpp"

namespace opencv_test { namespace {

/**
 * Verifies that assigning `nullptr` and a nonzero value to a `cvv::util::ObserverPtr<Int>` 
 * (from /src/util/observer_ptr.hpp) work and that `isNull()` and `getPtr()` return the correct result.
 */

using cvv::util::ObserverPtr;

TEST(ObserverPtrTest, ConstructionAssignment)
{
	ObserverPtr<int> ptr = nullptr;
	EXPECT_TRUE(ptr.isNull());
	int x = 3;
	ptr = x;
	EXPECT_FALSE(ptr.isNull());
	EXPECT_EQ(&x, ptr.getPtr());
}

}} // namespace