// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_CUVID_FFMPEG_API_H__
#define __OPENCV_CUVID_FFMPEG_API_H__
#include "../src/cap_ffmpeg_legacy_api.hpp"

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef OPENCV_FFMPEG_API
#if defined(__OPENCV_BUILD)
#   define OPENCV_FFMPEG_API
#elif defined _WIN32
#   define OPENCV_FFMPEG_API __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#   define OPENCV_FFMPEG_API __attribute__ ((visibility ("default")))
#else
#   define OPENCV_FFMPEG_API
#endif
#endif

/*
* For CUDA encoder
*/

OPENCV_FFMPEG_API struct OutputMediaStream_FFMPEG* create_OutputMediaStream_FFMPEG(const char* fileName, int width, int height, double fps);
OPENCV_FFMPEG_API void release_OutputMediaStream_FFMPEG(struct OutputMediaStream_FFMPEG* stream);
OPENCV_FFMPEG_API void write_OutputMediaStream_FFMPEG(struct OutputMediaStream_FFMPEG* stream, unsigned char* data, int size, int keyFrame);

typedef struct OutputMediaStream_FFMPEG* (*Create_OutputMediaStream_FFMPEG_Plugin)(const char* fileName, int width, int height, double fps);
typedef void (*Release_OutputMediaStream_FFMPEG_Plugin)(struct OutputMediaStream_FFMPEG* stream);
typedef void (*Write_OutputMediaStream_FFMPEG_Plugin)(struct OutputMediaStream_FFMPEG* stream, unsigned char* data, int size, int keyFrame);

/*
* For CUDA decoder
*/

OPENCV_FFMPEG_API struct InputMediaStream_FFMPEG* create_InputMediaStream_FFMPEG(const char* fileName, int* codec, int* chroma_format, int* width, int* height);
OPENCV_FFMPEG_API void release_InputMediaStream_FFMPEG(struct InputMediaStream_FFMPEG* stream);
OPENCV_FFMPEG_API int read_InputMediaStream_FFMPEG(struct InputMediaStream_FFMPEG* stream, unsigned char** data, int* size, int* endOfFile);

typedef struct InputMediaStream_FFMPEG* (*Create_InputMediaStream_FFMPEG_Plugin)(const char* fileName, int* codec, int* chroma_format, int* width, int* height);
typedef void (*Release_InputMediaStream_FFMPEG_Plugin)(struct InputMediaStream_FFMPEG* stream);
typedef int (*Read_InputMediaStream_FFMPEG_Plugin)(struct InputMediaStream_FFMPEG* stream, unsigned char** data, int* size, int* endOfFile);

#ifdef __cplusplus
}
#endif

#endif
