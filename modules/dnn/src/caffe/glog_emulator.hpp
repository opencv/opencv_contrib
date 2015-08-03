#ifndef __OPENCV_DNN_CAFFE_GLOG_EMULATOR__
#define __OPENCV_DNN_CAFFE_GLOG_EMULATOR__
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <opencv2/core.hpp>

#define CHECK(cond)     cv::GLogWrapper(__FILE__, CV_Func, __LINE__, "CHECK", #cond, cond)
#define CHECK_EQ(a, b)  cv::GLogWrapper(__FILE__, CV_Func, __LINE__, "CHECK", #a"="#b, ((a) == (b)))
#define LOG(TYPE)       cv::GLogWrapper(__FILE__, CV_Func, __LINE__, #TYPE)

namespace cv
{

class GLogWrapper
{
    std::stringstream stream;
    const char *file, *func, *type, *cond_str;
    int line;
    bool cond_staus;

public:

    GLogWrapper(const char *_file, const char *_func, int _line,
                const char *_type,
                const char *_cond_str = NULL, bool _cond_status = true
               ) :
               file(_file), func(_func), type(_type), cond_str(_cond_str),
               line(_line), cond_staus(_cond_status) {}

    template<typename T>
    GLogWrapper &operator<<(const T &v)
    {
        if (!cond_str || cond_str && !cond_staus)
            stream << v;
        return *this;
    }

    ~GLogWrapper()
    {
        if (cond_str && !cond_staus)
        {
            cv::error(cv::Error::StsError, "FAILED: " + String(cond_str) + "." + stream.str(), func, file, line);
        }
        else if (!cond_str && strcmp(type, "CHECK"))
        {
            if (!strcmp(type, "INFO"))
                std::cout << stream.str();
            else
                std::cerr << stream.str();
        }
    }
};

}
#endif