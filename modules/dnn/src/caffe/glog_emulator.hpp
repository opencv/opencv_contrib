#pragma once
#include <stdlib.h>
#include <iostream>
#include <opencv2/core.hpp>

#define CHECK(cond)     cv::GLogWrapper(__FILE__, CV_Func, __LINE__, "CHECK", #cond, cond)
#define CHECK_EQ(a, b)  cv::GLogWrapper(__FILE__, CV_Func, __LINE__, "CHECK", #a #b, ((a) == (b)))
#define LOG(TYPE)       cv::GLogWrapper(__FILE__, CV_Func, __LINE__, #TYPE)

namespace cv
{

class GLogWrapper
{
    const char *type, *cond_str, *file, *func;
    int line;
    bool cond_staus;
    std::ostream &stream;

    static std::ostream &selectStream(const char *type)
    {
        if (!strcmp(type, "INFO"))
            return std::cout;
        else
            return std::cerr;
    }

public:

    GLogWrapper(const char *_file, const char *_func, int _line, 
                const char *_type, 
                const char *_cond_str = NULL, bool _cond_status = true
               ) :
               stream(selectStream(_type)), 
               file(_file), func(_func), line(_line), 
               type(_type), cond_str(_cond_str), cond_staus(_cond_status) {}

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
            cv::error(cv::Error::StsAssert, cond_str, func, file, line);
        }
        //else if (!cond_str && strcmp(type, "INFO"))
        //{
        //    cv::error(cv::Error::StsAssert, type, func, file, line);
        //}
    }
};

}