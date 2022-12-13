#ifndef SRC_COMMON_UTIL_HPP_
#define SRC_COMMON_UTIL_HPP_

#include <string>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

namespace kb {
namespace viz2d {

void gl_check_error(const std::filesystem::path &file, unsigned int line, const char *expression);

#define GL_CHECK(expr)                            \
    expr;                                        \
    kb::viz2d::gl_check_error(__FILE__, __LINE__, #expr);

void error_callback(int error, const char *description);
std::string get_gl_info();
std::string get_cl_info();
void print_system_info();
void update_fps(cv::Ptr<Viz2D> viz2d, bool graphical);
}
}

#endif /* SRC_COMMON_UTIL_HPP_ */
