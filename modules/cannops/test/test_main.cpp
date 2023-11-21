// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

class CannEnvironment : public ::testing::Environment
{
public:
    virtual ~CannEnvironment() = default;
    virtual void SetUp() CV_OVERRIDE { initAcl(); }
    virtual void TearDown() CV_OVERRIDE { finalizeAcl(); }
};

static void initTests()
{
    CannEnvironment* cannEnv = new CannEnvironment();
    ::testing::AddGlobalTestEnvironment(cannEnv);
}

CV_TEST_MAIN("cannops", initTests());
