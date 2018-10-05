// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "proto_terms.hpp"

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iostream>
namespace cv { namespace pb {

ProtoEnum::ProtoEnum(bool _packed, const std::string& defaultValue)
    : ProtoValue(PB_STRING, _packed, defaultValue)
{

}

void ProtoEnum::addValue(const std::string& name, int number)
{
    std::pair<int, std::string> enumValue(number, name);
    CV_Assert(enumValues.insert(enumValue).second);
}

void ProtoEnum::read(std::istream& s)
{
    ProtoValue ids(PB_INT32, packed);
    ids.read(s);

    std::map<int, std::string>::iterator it;
    for (int i = 0; i < (int)ids.size(); ++i)
    {
        int id = ids.getInt32(i);
        it = enumValues.find(id);
        CV_Assert(it != enumValues.end());

        std::vector<std::string> value(1, it->second);
        std::vector<std::string>::iterator valueIt = value.begin();
        ProtoValue::read(valueIt);
    }
}

Ptr<ProtobufField> ProtoEnum::clone() const
{
    Ptr<ProtoEnum> copy(new ProtoEnum(packed, defaultValueStr));
    copy->enumValues = enumValues;
    return copy;
}

}  // namespace pb
}  // namespace cv
