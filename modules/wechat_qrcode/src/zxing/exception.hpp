// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_EXCEPTION_HPP__
#define __ZXING_EXCEPTION_HPP__

#include <exception>
#include <string>

namespace zxing {

class Exception : public std::exception {
private:
    char const* const message;

public:
    Exception() throw() : message(0) {}
    explicit Exception(const char* msg) throw() : message(copy(msg)) {}
    Exception(Exception const& that) throw() : std::exception(that), message(copy(that.message)) {}
    ~Exception() throw() {
        if (message) {
            deleteMessage();
        }
    }
    char const* what() const throw() override { return message ? message : ""; }

private:
    static char const* copy(char const*);
    void deleteMessage();
};

}  // namespace zxing

#endif  // __ZXING_EXCEPTION_HPP__
