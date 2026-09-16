/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// rocDecode <-> NVCUVID compatibility layer.
//
// The cudacodec VideoReader decode pipeline (frame queue, video source, video
// parser, video decoder, video reader) is written against the NVIDIA NVCUVID /
// cuvid API. rocDecode provides the same decode model (a video parser that calls
// back into a hardware decoder, then a post-process/map step that hands back YUV
// device surfaces) under different names. This header maps the small set of
// cuvid type names and helper routines the shared cudacodec code uses onto their
// rocDecode equivalents so that the codec-agnostic plumbing compiles unchanged
// and only the decoder/parser back ends (video_decoder.cpp, video_parser.cpp)
// differ. It is included only on the ROCm/HIP path (HAVE_ROCDECODE).
//
// \author Jeff Daily <jeff.daily@amd.com>

#ifndef __CUDACODEC_ROCDECODE_VIDEO_COMPAT_HPP__
#define __CUDACODEC_ROCDECODE_VIDEO_COMPAT_HPP__

#include <hip/hip_runtime.h>
#include "rocdecode/rocdecode.h"
#include "rocdecode/rocparser.h"

namespace cv { namespace cudacodec { namespace detail {

// Display-order frame descriptor delivered by the parser. Mirrors the subset of
// CUVIDPARSERDISPINFO that the cudacodec plumbing reads; field names are kept
// identical to RocdecParserDispInfo so the parser callback can fill it directly.
struct RocdecDispInfo
{
    int picture_index = 0;
    int progressive_frame = 0;
    int top_field_first = 0;
    int repeat_first_field = 0;
    int64_t pts = 0;
};

// Per-display post-process parameters. Mirrors the subset of CUVIDPROCPARAMS the
// reader sets before mapping a frame; carries the rocDecProcParams plus the
// caller's HIP stream so the map/convert step can run on it.
struct RocdecProcInfo
{
    int progressive_frame = 0;
    int second_field = 0;
    int top_field_first = 0;
    int unpaired_field = 0;
    hipStream_t output_stream = 0;
    unsigned long long* histogram_dptr = 0; // unused on ROCm (no luma histogram)
};

}}} // namespace cv::cudacodec::detail

// The shared plumbing (frame_queue, video_source, video_reader) refers to the
// frame and proc descriptors by their cuvid names; map them to the rocDecode
// compat structs so that code stays back-end neutral.
typedef cv::cudacodec::detail::RocdecDispInfo CUVIDPARSERDISPINFO;
typedef cv::cudacodec::detail::RocdecProcInfo CUVIDPROCPARAMS;

// Context-lock and HIP handle spellings used by the reader. ROCm has no
// per-context video lock (rocDecode is internally thread-safe and bound to a
// device, not a driver context), so the lock degenerates to a no-op handle and
// the "context" is just the active HIP device id.
typedef int CUvideoctxlock;
typedef int CUcontext;
// Driver device-pointer / stream spellings the reader still names on the (dead on
// ROCm) luma-histogram path; map to their HIP equivalents so it compiles.
typedef void* CUdeviceptr;
typedef hipStream_t CUstream;

// Driver-context plumbing the reader inherits from the cuvid path. rocDecode
// binds to a HIP device rather than a driver context and needs no explicit
// lock, so these reduce to recording the active device / no-ops. They keep the
// reader source identical between the two back ends.
static inline int rocdecCtxGetCurrent(CUcontext* ctx) { hipGetDevice(ctx); return 0; }
static inline int rocdecCtxLockCreate(CUvideoctxlock* lock, CUcontext) { *lock = 0; return 0; }
static inline int rocdecCtxLock(CUvideoctxlock, unsigned int) { return 0; }
static inline int rocdecCtxUnlock(CUvideoctxlock, unsigned int) { return 0; }

#define cuCtxGetCurrent      rocdecCtxGetCurrent
#define cuvidCtxLockCreate   rocdecCtxLockCreate
#define cuvidCtxLock         rocdecCtxLock
#define cuvidCtxUnlock       rocdecCtxUnlock

// The reader wraps these driver calls in cuSafeCall (a check macro). On ROCm the
// helpers above always succeed, so cuSafeCall is just evaluation. The histogram
// device-copy (cuMemcpyDtoDAsync) is unreachable on ROCm because rocDecode has no
// luma histogram output (enableHistogram is rejected at caps-query time), but it
// must still compile; map it to the HIP async copy.
#ifndef cuSafeCall
#define cuSafeCall(expr) (expr)
#endif
static inline hipError_t rocdecMemcpyDtoDAsync(void* dst, unsigned long long src, size_t n, hipStream_t s) {
    return hipMemcpyDtoDAsync(dst, reinterpret_cast<void*>(src), n, s);
}
#define cuMemcpyDtoDAsync rocdecMemcpyDtoDAsync

#endif // __CUDACODEC_ROCDECODE_VIDEO_COMPAT_HPP__
