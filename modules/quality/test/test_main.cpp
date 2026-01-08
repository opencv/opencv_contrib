// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

CV_TEST_MAIN("",
    cvtest::addDataSearchSubDirectory("contrib/quality")    // for ocv_add_testdata
    , cvtest::addDataSearchSubDirectory("quality")          // for ${OPENCV_TEST_DATA_PATH}
)