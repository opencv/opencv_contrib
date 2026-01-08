// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#include "../precomp.hpp"
#include "errorhandler.hpp"

namespace zxing {

ErrorHandler::ErrorHandler() : err_code_(0), err_msg_("") { Init(); }

ErrorHandler::ErrorHandler(const char* err_msg) : err_code_(-1), err_msg_(err_msg) { Init(); }

ErrorHandler::ErrorHandler(std::string& err_msg) : err_code_(-1), err_msg_(err_msg) { Init(); }

ErrorHandler::ErrorHandler(int err_code) : err_code_(err_code), err_msg_("error") { Init(); }

ErrorHandler::ErrorHandler(int err_code, const char* err_msg)
    : err_code_(err_code), err_msg_(err_msg) {
    Init();
}

ErrorHandler::ErrorHandler(const ErrorHandler& other) {
    err_code_ = other.ErrCode();
    err_msg_.assign(other.ErrMsg());
    Init();
}

ErrorHandler& ErrorHandler::operator=(const ErrorHandler& other) {
    err_code_ = other.ErrCode();
    err_msg_.assign(other.ErrMsg());
    Init();
    return *this;
}

void ErrorHandler::Init() { handler_type_ = KErrorHandler; }

void ErrorHandler::Reset() {
    err_code_ = 0;
    err_msg_.assign("");
}

void ErrorHandler::PrintInfo() {
    printf("handler_tpye %d, error code %d, errmsg %s\n", handler_type_, err_code_,
           err_msg_.c_str());
}
}  // namespace zxing
