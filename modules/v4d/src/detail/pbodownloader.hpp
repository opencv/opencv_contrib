#ifndef POLY_PBO_DOWNLOADER_H
#define POLY_PBO_DOWNLOADER_H

#define __STDC_LIMIT_MACROS
#include "opencv2/v4d/v4d.hpp"
#include <stdint.h>

#ifdef OPENCV_V4D_USE_ES3
#define GLFW_INCLUDE_ES3
#define GLFW_INCLUDE_GLEXT
#endif

#include <GLFW/glfw3.h>

namespace poly {

  class PboDownloader {
  public:
    PboDownloader();
    ~PboDownloader();
    int init(GLenum fmt, int w, int h, int num);
    void download();

  public:
    GLenum fmt;
    GLuint* pbos;
    uint64_t num_pbos;
    uint64_t dx;
    uint64_t num_downloads;
    int width;
    int height;
    int nbytes; /* number of bytes in the pbo buffer. */
    unsigned char* pixels; /* the downloaded pixels. */
  };

} /* namespace poly */

#endif
