// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>


#ifndef MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_GL_HPP_
#define MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_GL_HPP_

#  if !defined(OPENCV_V4D_USE_ES3)
#    include "GL/glew.h"
#    define GLFW_INCLUDE_NONE
#  else
#    define GLFW_INCLUDE_ES3
#    define GLFW_INCLUDE_GLEXT
#  endif
#  include <GLFW/glfw3.h>

#endif /* MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_GL_HPP_ */
