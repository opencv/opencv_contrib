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

// Read <numBytes> bytes into <dst> from <s> binary stream.
// Returns number of bytes that were exactly read.
static int readBinary(std::istream& s, void* dst, int numBytes)
{
    s.read((char*)dst, numBytes);
    CV_Assert(s.gcount() == numBytes || s.gcount() == 0);
    return (int)s.gcount();
}

static int readVarint(std::istream& s, void* dst, int maxNumBytes)
{
    CV_Assert(0 <= maxNumBytes && maxNumBytes <= 8);
    uint64_t res = 0;
    char byte;
    bool read_next_byte = (readBinary(s, &byte, 1) != 0);
    int bytesRead = 0;
    // Read bytes until the first bit of byte is zero.
    // Maximal length - 9 bytes (7 bits from every byte , 63 bits totally).
    for (; bytesRead < 9 && read_next_byte; ++bytesRead)
    {
        read_next_byte = (byte & 0x80) != 0;
        uint64_t mask = (byte & 0x7f);
        res |= mask << bytesRead * 7;  // All bits except the last one.

        if (read_next_byte && !readBinary(s, &byte, 1))
        {
            CV_Error(Error::StsParseError, "Unexpected end of file");
        }
    }
    if (read_next_byte)
    {
        bytesRead += 1;
        read_next_byte = (byte & 0x80) != 0;
        CV_Assert(!read_next_byte);
    }
    memcpy(dst, &res, maxNumBytes);
    return bytesRead;
}

template <typename T>
static int getProtoType()
{
    if (typeid(T) == typeid(int32_t))
        return PB_INT32;
    if (typeid(T) == typeid(uint32_t))
        return PB_UINT32;
    if (typeid(T) == typeid(int64_t))
        return PB_INT64;
    if (typeid(T) == typeid(uint64_t))
        return PB_UINT64;
    if (typeid(T) == typeid(bool))
        return PB_BOOL;
    if (typeid(T) == typeid(float))
        return PB_FLOAT;
    if (typeid(T) == typeid(double))
        return PB_DOUBLE;
    if (typeid(T) == typeid(std::string))
        return PB_STRING;
    CV_Error(Error::StsNotImplemented, "Unknown type id");
    return -1;
}

template <typename T>
static T valueFromString(const std::string& str)
{
    T value = 0;
    if (!str.empty())
    {
        if (typeid(T) != typeid(bool))
        {
            std::stringstream ss(str);
            ss >> value;
        }
        else if (str == "true")
        {
            memset(&value, true, 1);
        }
        else if (str == "false")
        {
            memset(&value, false, 1);
        }
        else
        {
            CV_Error(Error::StsParseError,
                     "Cannot interpret boolean value: " + str);
        }
    }
    return value;
}

ProtoValue::ProtoValue(int t, bool _packed, const std::string& defaultValue)
    : ProtobufField(t)
{
    packed = _packed;
    defaultValueStr = defaultValue;
}

ProtoValue::ProtoValue(int t, std::istream& s) : ProtobufField(t)
{
    packed = false;
    defaultValueStr = "";
    read(s);
}

size_t ProtoValue::elemSize(int idx) const
{
    if (_type == PB_INT32 || _type == PB_UINT32 || _type == PB_FLOAT)
        return 4;
    else if (_type == PB_INT64 || _type == PB_UINT64 || _type == PB_DOUBLE)
        return 8;
    else if (_type == PB_BOOL)
        return 1;
    else if (_type == PB_STRING)
        return (idx != (int)offsets.size() - 1 ? offsets[idx + 1] : data.size()) - offsets[idx];
    else
        CV_Error(Error::StsNotImplemented, format("Could not determine a size of type [%d]", _type));
    return 0;
}

void ProtoValue::read(std::istream& s)
{
    if (_type == PB_INT32 || _type == PB_INT64 || _type == PB_UINT32 || _type == PB_UINT64)
    {
        const int bytesPerValue = (int)elemSize();
        if (packed)
        {
            int numBytes = ProtoValue(PB_INT32, s).get<int32_t>();
            int end = (int)s.tellg() + numBytes;
            while (s.tellg() < end)
            {
                offsets.push_back(data.size());
                data.resize(data.size() + bytesPerValue);
                readVarint(s, &data[offsets.back()], bytesPerValue);
            }
            CV_Assert((int)s.tellg() == end);
        }
        else
        {
            offsets.push_back(data.size());
            data.resize(data.size() + bytesPerValue);
            readVarint(s, &data[offsets.back()], bytesPerValue);
        }
    }
    else if (_type == PB_FLOAT || _type == PB_DOUBLE || _type == PB_BOOL)
    {
        const int bytesPerValue = (int)elemSize();
        if (packed)
        {
            int numBytes = ProtoValue(PB_INT32, s).get<int32_t>();
            CV_Assert(numBytes > 0, numBytes % bytesPerValue == 0);

            for (int i = 0; i < numBytes; i += bytesPerValue)
            {
                offsets.push_back(data.size() + i);
            }
            data.resize(data.size() + numBytes);
            CV_Assert(readBinary(s, &data[data.size() - numBytes], numBytes));
        }
        else
        {
            offsets.push_back(data.size());
            data.resize(data.size() + bytesPerValue);
            CV_Assert(readBinary(s, &data[offsets.back()], bytesPerValue));
        }
    }
    else if (_type == PB_STRING)
    {
        int len = ProtoValue(PB_INT32, s).get<int32_t>();
        if (len < 0)
            CV_Error(Error::StsParseError, "Negative string length");
        if (len != 0)
        {
            offsets.push_back(data.size());
            data.resize(data.size() + len);
            CV_Assert(readBinary(s, &data[offsets.back()], len));
        }
    }
    else
        CV_Error(Error::StsNotImplemented, "Unsupported protobuf value type");
}

void ProtoValue::read(std::vector<std::string>::iterator& tokenIt)
{
    std::string str = *tokenIt;
    ++tokenIt;

    offsets.push_back(data.size());
    if (_type != PB_STRING)
    {
        data.resize(data.size() + elemSize());
        void* ptr = &data[offsets.back()];
        switch (_type)
        {
            case PB_INT32:  *(int32_t*)ptr = valueFromString<int32_t>(str); break;
            case PB_UINT32: *(uint32_t*)ptr = valueFromString<uint32_t>(str); break;
            case PB_INT64:  *(int64_t*)ptr = valueFromString<int64_t>(str); break;
            case PB_UINT64: *(uint64_t*)ptr = valueFromString<uint64_t>(str); break;
            case PB_FLOAT:  *(float*)ptr = valueFromString<float>(str); break;
            case PB_DOUBLE: *(double*)ptr = valueFromString<double>(str); break;
            case PB_BOOL:   *(bool*)ptr = valueFromString<bool>(str); break;
            default:
                CV_Error(Error::StsParseError, format("Unknown data type [%d]", _type));
        }
    }
    else
    {
        data.resize(data.size() + str.size());
        memcpy(&data[offsets.back()], &str[0], str.size());
    }
}

Ptr<ProtobufField> ProtoValue::clone() const
{
    return Ptr<ProtobufField>(new ProtoValue(_type, packed, defaultValueStr));
}

Ptr<ProtobufField> ProtoValue::operator[](int idx) const
{
    Ptr<ProtoValue> p(new ProtoValue(_type, packed, defaultValueStr));
    if (!data.empty())
    {
        if (idx < 0 || idx >= (int)offsets.size())
            CV_Error(Error::StsOutOfRange, format("Index [%d] out of range [0, %d)", idx, offsets.size()));
        const char* value = &data[offsets[idx]];
        p->data = std::vector<char>(value, value + elemSize(idx));
        p->offsets = std::vector<size_t>(1, 0);
    }
    return p;
}

void ProtoValue::clear()
{
    data.clear();
    offsets.clear();
}

size_t ProtoValue::size() const
{
    return offsets.size();
}

bool ProtoValue::empty() const
{
    return offsets.empty();
}

void ProtoValue::copyTo(void* dst, int numBytes) const
{
    CV_Assert((size_t)numBytes == data.size());
    memcpy(dst, &data[0], numBytes);
}

int32_t ProtoValue::getInt32(int idx) const { return get<int32_t>(idx); }
uint32_t ProtoValue::getUInt32(int idx) const { return get<uint32_t>(idx); }
int64_t ProtoValue::getInt64(int idx) const { return get<int64_t>(idx); }
uint64_t ProtoValue::getUInt64(int idx) const { return get<uint64_t>(idx); }
float ProtoValue::getFloat(int idx) const { return get<float>(idx); }
double ProtoValue::getDouble(int idx) const { return get<double>(idx); }
bool ProtoValue::getBool(int idx) const { return get<bool>(idx); }

template <typename T>
T ProtoValue::get(int idx) const
{
    if (getProtoType<T>() != _type)
        CV_Error(Error::StsUnmatchedFormats,
                 format("Type mismatch: source [%d] and destination [%d]", _type, getProtoType<T>()));
    if (!data.empty())
    {
        if (idx < 0 || (size_t)idx >= offsets.size())
            CV_Error(Error::StsOutOfRange, format("Index [%d] out of range [0, %d)", idx, offsets.size()));
        T dst;
        memcpy(&dst, &data[offsets[idx]], sizeof(T));
        return dst;
    }
    return valueFromString<T>(defaultValueStr);
}

template <typename T>
void ProtoValue::set(const std::string& valueStr, int idx)
{
    if (getProtoType<T>() != _type)
        CV_Error(Error::StsUnmatchedFormats,
                 format("Type mismatch: source [%d] and destination [%d]", _type, getProtoType<T>()));
    CV_Assert(0 <= idx && idx < (int)offsets.size());
    T src = valueFromString<T>(valueStr);
    memcpy(&data[offsets[idx]], &src, sizeof(T));
}

std::string ProtoValue::getString(int idx) const
{
    if (_type != PB_STRING)
        CV_Error(Error::StsUnmatchedFormats,
                 format("Type mismatch: source [%d] and destination [%d]", _type, PB_STRING));
    if (!data.empty())
    {
        if (idx < 0 || (size_t)idx >= offsets.size())
            CV_Error(Error::StsOutOfRange, format("Index [%d] out of range [0, %d)", idx, offsets.size()));
        const char* ptr = &data[offsets[idx]];
        return std::string(ptr, ptr + elemSize(idx));
    }
    return defaultValueStr;
}

void ProtoValue::set(const std::string& valueStr, int idx)
{
    switch (_type)
    {
        case PB_INT32:  set<int32_t>(valueStr, idx); break;
        case PB_UINT32: set<uint32_t>(valueStr, idx); break;
        case PB_INT64:  set<int64_t>(valueStr, idx); break;
        case PB_UINT64: set<uint64_t>(valueStr, idx); break;
        case PB_FLOAT:  set<float>(valueStr, idx); break;
        case PB_DOUBLE: set<double>(valueStr, idx); break;
        case PB_BOOL:   set<bool>(valueStr, idx); break;
        case PB_STRING:
        {
            CV_Assert(0 <= idx && idx < (int)offsets.size());
            size_t numBytes = elemSize(idx);
            data.erase(data.begin() + offsets[idx], data.begin() + offsets[idx] + numBytes);
            data.insert(data.begin() + offsets[idx], valueStr.begin(), valueStr.end());
            for (int i = idx + 1; i < (int)offsets.size(); ++i)
            {
                offsets[i] += (int)valueStr.size() - numBytes;
            }
            break;
        }
        default:
            CV_Error(Error::StsParseError, format("Unknown data type [%d]", _type));
    }
}

}  // namespace pb
}  // namespace cv
