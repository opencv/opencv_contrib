// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_CHARACTERSETECI_HPP__
#define __ZXING_COMMON_CHARACTERSETECI_HPP__

#include <map>
#include "../decodehints.hpp"
#include "counted.hpp"

namespace zxing {
namespace common {

class CharacterSetECI : public Counted {
private:
    static std::map<int, zxing::Ref<CharacterSetECI> > VALUE_TO_ECI;
    static std::map<std::string, zxing::Ref<CharacterSetECI> > NAME_TO_ECI;
    static const bool inited;
    static bool init_tables();

    int const* const values_;
    char const* const* const names_;

    CharacterSetECI(int const* values, char const* const* names);

    static void addCharacterSet(int const* value, char const* const* encodingNames);

public:
    char const* name() const;
    int getValue() const;

    static CharacterSetECI* getCharacterSetECIByValueFind(int value);
    static CharacterSetECI* getCharacterSetECIByName(std::string const& name);
};

}  // namespace common
}  // namespace zxing

#endif  // __ZXING_COMMON_CHARACTERSETECI_HPP__
