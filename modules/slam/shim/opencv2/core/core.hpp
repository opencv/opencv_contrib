#pragma once

// This is a build-only shim for third-party libraries that still include
// <opencv2/core/core.hpp> (legacy compatibility header).
//
// When building OpenCV itself, the real <opencv2/core/core.hpp> intentionally
// triggers an #error to prevent internal use of compatibility headers.
//
// We provide this shim via a PRIVATE include directory (added BEFORE others) so
// external deps like DBoW3 can compile while we are inside the OpenCV build.
// It is NOT installed and does not affect OpenCV's public headers.

#include <opencv2/core.hpp>
