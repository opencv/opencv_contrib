// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_STR_HPP__
#define __ZXING_COMMON_STR_HPP__

#include "counted.hpp"

#include <sstream>
#include <string>
using std::string;
namespace zxing {

class String : public Counted {
private:
    std::string text_;

public:
    explicit String(const std::string& text);
    explicit String(int);
    char charAt(int) const;
    Ref<String> substring(int) const;
    Ref<String> substring(int, int) const;
    const std::string& getText() const;
    int size() const;
    void append(std::string const& tail);
    void append(char c);
    void append(int d);
    void append(Ref<String> str);
    int length() const;
};

class StrUtil {
public:
    static string COMBINE_STRING(string str1, string str2);
    static string COMBINE_STRING(string str1, char c);
    static string COMBINE_STRING(string str1, int d);
    static Ref<String> COMBINE_STRING(char c1, Ref<String> content, char c2);

    template <typename T>
    static string numberToString(T Number);

    template <typename T>
    static T stringToNumber(const string& Text);

    static int indexOf(const char* str, char c);
};

}  // namespace zxing

#endif  // __ZXING_COMMON_STR_HPP__
