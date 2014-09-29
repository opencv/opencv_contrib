#ifndef __OPENCV_PHOTOEFFECTS_PRECOMP_HPP__
#define __OPENCV_PHOTOEFFECTS_PRECOMP_HPP__

#include <opencv2/photoeffects.hpp>
#include <opencv2/core/core.hpp>

#include <gtest/gtest.h>

int suppressAssertionMessage(int, const char *, const char *,
                            const char *, int, void *);


#define EXPECT_ERROR(expectedErrorCode, expr) \
{\
    cv::setBreakOnError(false);\
    cv::redirectError(suppressAssertionMessage);\
    int errorCode = 0;\
    try\
    {\
        (expr);\
    }\
    catch (cv::Exception & e)\
    {\
        errorCode = e.code;\
    }\
    cv::setBreakOnError(true);\
    cv::redirectError(0);\
    EXPECT_EQ((int)(expectedErrorCode), errorCode);\
}

#endif