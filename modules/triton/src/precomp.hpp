// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright 2026, The FlagOS Contributors.
#ifndef OPENCV_TRITON_PRECOMP_HPP
#define OPENCV_TRITON_PRECOMP_HPP
#include "core.hpp"
#include "opencv2/core/private.hpp"
#include "opencv2/imgproc.hpp"   // INTER_LINEAR / INTER_NEAREST -- needed by
// core_resize.cpp, not just the CUDA adapter.

// The CUDA stream accessor includes cuda_runtime.h. Keep it out of HIP builds;
// CUDA adapters use it to unwrap the public OpenCV stream.
#ifdef OPENCV_TRITON_HAVE_CUDA_ADAPTER
#include "opencv2/triton.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#endif
#endif
