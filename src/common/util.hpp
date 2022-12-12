#ifndef SRC_COMMON_UTIL_HPP_
#define SRC_COMMON_UTIL_HPP_

#include <string>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

namespace kb {

class GLWindow;

typedef cv::ocl::OpenCLExecutionContext CLExecContext_t;
typedef cv::ocl::OpenCLExecutionContextScope CLExecScope_t;

void error_callback(int error, const char *description);
std::string get_gl_info();
std::string get_cl_info();
void print_system_info();
void update_fps(cv::Ptr<GLWindow> window, bool graphical = false);
} //namespace kb

#endif /* SRC_COMMON_UTIL_HPP_ */
