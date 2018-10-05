// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <string>
#include <iostream>
#include "proto_terms.hpp"

namespace cv { namespace pb {

static Ptr<ProtoValue> toValue(const ProtobufFields& fields)
{
    CV_Assert(fields.size() == 1);
    Ptr<ProtoValue> p = fields[0].dynamicCast<ProtoValue>();
    CV_Assert(!p.empty());
    return p;
}

static Ptr<ProtoMessage> toMessage(const ProtobufFields& fields)
{
    CV_Assert(fields.size() == 1);
    Ptr<ProtoMessage> p = fields[0].dynamicCast<ProtoMessage>();
    CV_Assert(!p.empty());
    return p;
}

ProtobufNode::ProtobufNode(const ProtobufFields& _fields) : fields(_fields)
{

}

ProtobufNode::ProtobufNode(const Ptr<ProtobufField>& field) : fields(ProtobufFields(1, field))
{

}

ProtobufNode ProtobufNode::operator[](int idx) const
{
    if (type() == PB_MESSAGE)
    {
        CV_Assert(0 <= idx && idx < (int)fields.size());
        return ProtobufNode(fields[idx]);
    }
    return ProtobufNode(toValue(fields)->operator[](idx));
}

bool ProtobufNode::empty() const
{
    for (size_t i = 0; i < fields.size(); ++i)
    {
        if (!fields[i]->empty())
            return false;
    }
    return true;
}

size_t ProtobufNode::size() const
{
    if (empty())
        return 0;

    if (type() == PB_MESSAGE)
    {
        return fields.size();
    }
    CV_Assert(fields.size() == 1);
    return fields[0].dynamicCast<ProtoValue>()->size();
}

bool ProtobufNode::has(const std::string& name) const
{
    return toMessage(fields)->has(name);
}

void ProtobufNode::remove(const std::string& name, int idx)
{
    return toMessage(fields)->remove(name, idx);
}

ProtobufNode ProtobufNode::operator[](const std::string& name) const
{
    return ProtobufNode(toMessage(fields)->operator[](name));
}

ProtobufNode ProtobufNode::operator[](const char* name) const
{
    return operator[](std::string(name));
}

int ProtobufNode::type() const
{
    CV_Assert(!fields.empty());
    int t = fields[0]->type();
    for (size_t i = 1; i < fields.size(); ++i)
        CV_Assert(fields[i]->type() == t);
    return t;
}

void ProtobufNode::copyTo(int numBytes, void* dst) const
{
    toValue(fields)->copyTo(dst, numBytes);
}

void ProtobufNode::operator >> (int32_t& value) const { value = toValue(fields)->getInt32(); }
void ProtobufNode::operator >> (uint32_t& value) const { value = toValue(fields)->getUInt32(); }
void ProtobufNode::operator >> (int64_t& value) const { value = toValue(fields)->getInt64(); }
void ProtobufNode::operator >> (uint64_t& value) const { value = toValue(fields)->getUInt64(); }
void ProtobufNode::operator >> (float& value) const { value = toValue(fields)->getFloat(); }
void ProtobufNode::operator >> (double& value) const { value = toValue(fields)->getDouble(); }
void ProtobufNode::operator >> (bool& value) const { value = toValue(fields)->getBool(); }
void ProtobufNode::operator >> (std::string& str) const { str = toValue(fields)->getString(); }

ProtobufNode::operator int32_t() const { return toValue(fields)->getInt32(); }
ProtobufNode::operator int64_t() const { return toValue(fields)->getInt64(); }
ProtobufNode::operator uint32_t() const { return toValue(fields)->getUInt32(); }
ProtobufNode::operator uint64_t() const { return toValue(fields)->getUInt64(); }
ProtobufNode::operator float() const { return toValue(fields)->getFloat(); }
ProtobufNode::operator double() const { return toValue(fields)->getDouble(); }
ProtobufNode::operator bool() const { return toValue(fields)->getBool(); }
ProtobufNode::operator std::string() const { return toValue(fields)->getString(); }

void ProtobufNode::set(const std::string& str, int idx)
{
    toValue(fields)->set(str, idx);
}

std::vector<std::string> ProtobufNode::readFields() const
{
    if (type() == PB_MESSAGE)
    {
        CV_Assert(fields.size() == 1);
        return fields[0].dynamicCast<ProtoMessage>()->readFields();
    }
    return std::vector<std::string>();
}

}  // namespace pb
}  // namespace cv
