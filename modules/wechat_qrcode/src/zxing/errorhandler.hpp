// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __ZXING_ERRORHANDLER_HPP__
#define __ZXING_ERRORHANDLER_HPP__

#include <string>

namespace zxing {

enum {
    KErrorHandler = 0,
    KErrorHandler_NotFound = 1,
    KErrorHandler_CheckSum = 2,
    KErrorHandler_Reader = 3,
    KErrorHandler_IllegalArgument = 4,
    KErrorHandler_ReedSolomon = 5,
    KErrorHandler_Format = 6,
    KErrorHandler_Detector = 7,
    KErrorHandler_IllegalState = 8,
};

class ErrorHandler {
public:
    ErrorHandler();
    explicit ErrorHandler(std::string& err_msg);
    explicit ErrorHandler(const char* err_msg);
    explicit ErrorHandler(int err_code);
    ErrorHandler(int err_code, std::string& err_msg);
    ErrorHandler(int err_code, const char* err_msg);

    virtual ~ErrorHandler(){};

    virtual inline int ErrCode() const { return err_code_; }
    virtual inline const std::string& ErrMsg() const { return err_msg_; }
    virtual inline int HandlerType() const { return handler_type_; }

    virtual void Init();
    ErrorHandler(const ErrorHandler& other);
    ErrorHandler& operator=(const ErrorHandler& other);

    virtual void PrintInfo();
    virtual void Reset();

protected:
    int handler_type_;

private:
    int err_code_;
    std::string err_msg_;
};

#define DECLARE_ERROR_HANDLER(__HANDLER__)                                                      \
    class __HANDLER__##ErrorHandler : public ErrorHandler {                                     \
    public:                                                                                     \
        __HANDLER__##ErrorHandler() : ErrorHandler() { Init(); };                               \
        __HANDLER__##ErrorHandler(std::string& err_msg) : ErrorHandler(err_msg) { Init(); };    \
        __HANDLER__##ErrorHandler(const char* err_msg) : ErrorHandler(err_msg) { Init(); };     \
        __HANDLER__##ErrorHandler(int err_code) : ErrorHandler(err_code) { Init(); };           \
        __HANDLER__##ErrorHandler(int err_code, std::string& err_msg)                           \
            : ErrorHandler(err_code, err_msg) {                                                 \
            Init();                                                                             \
        };                                                                                      \
        __HANDLER__##ErrorHandler(int err_code, const char* err_msg)                            \
            : ErrorHandler(err_code, err_msg) {                                                 \
            Init();                                                                             \
        };                                                                                      \
        __HANDLER__##ErrorHandler(const ErrorHandler& other) : ErrorHandler(other) { Init(); }; \
        void Init() override { handler_type_ = KErrorHandler_##__HANDLER__; }                            \
    };

DECLARE_ERROR_HANDLER(Reader)
DECLARE_ERROR_HANDLER(IllegalArgument)
DECLARE_ERROR_HANDLER(ReedSolomon)
DECLARE_ERROR_HANDLER(Format)
DECLARE_ERROR_HANDLER(Detector)
DECLARE_ERROR_HANDLER(NotFound)
DECLARE_ERROR_HANDLER(CheckSum)
DECLARE_ERROR_HANDLER(IllegalState)

#undef DECLARE_ERROR_HANDLER

}  // namespace zxing

#endif  // __ZXING_ERRORHANDLER_HPP__
