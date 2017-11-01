// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_PROTOBUF_PARSER_PROTO_TERMS_HPP__
#define __OPENCV_PROTOBUF_PARSER_PROTO_TERMS_HPP__

#include <map>
#include <vector>
#include <string>
#include <typeinfo>

#include "precomp.hpp"

namespace cv { namespace pb {

class ProtoValue : public ProtobufField
{
public:
    ProtoValue(int type, bool packed, const std::string& defaultValue = "");

    ProtoValue(int type, std::istream& s);

    virtual void read(std::istream& s);

    virtual void read(std::vector<std::string>::iterator& tokenIt);

    virtual Ptr<ProtobufField> clone() const;

    virtual void clear();

    virtual bool empty() const;

    virtual size_t size() const;

    void copyTo(void* dst, int numBytes) const;

    Ptr<ProtobufField> operator[](int idx) const;

    int32_t getInt32(int idx = 0) const;
    uint32_t getUInt32(int idx = 0) const;
    int64_t getInt64(int idx = 0) const;
    uint64_t getUInt64(int idx = 0) const;
    float getFloat(int idx = 0) const;
    double getDouble(int idx = 0) const;
    bool getBool(int idx = 0) const;
    std::string getString(int idx = 0) const;

    void set(const std::string& valueStr, int idx);

protected:
    bool packed;
    std::string defaultValueStr;

private:
    template <typename T>
    T get(int idx = 0) const;

    template <typename T>
    void set(const std::string& valueStr, int idx);

    // Returns size in bytes that takes specific value.
    // It's a fixed size for numerical size and length for strings.
    size_t elemSize(int idx = 0) const;

    // Raw read data.
    std::vector<char> data;
    // Offsets to values at data.
    std::vector<size_t> offsets;
};

class ProtoEnum : public ProtoValue
{
public:
    ProtoEnum(bool packed, const std::string& defaultValue);

    void addValue(const std::string& name, int number);

    virtual void read(std::istream& s);

    virtual Ptr<ProtobufField> clone() const;

private:
    std::map<int, std::string> enumValues;
};

// Structure that represents protobuf's message.
class ProtoMessage : public ProtobufField
{
public:
    ProtoMessage();

    void addField(const Ptr<ProtobufField>& field, const std::string& name, int tag);

    void addField(int type, const std::string& name, int tag);

    virtual void read(std::istream& s);

    virtual void read(std::vector<std::string>::iterator& tokenIt);

    virtual Ptr<ProtobufField> clone() const;

    virtual void clear();

    virtual bool empty() const;

    ProtobufNode operator[](const std::string& name) const;

    bool has(const std::string& name) const;

    void remove(const std::string& name, int idx = 0);

    // Returns names of fields were read in serialized message.
    std::vector<std::string> readFields() const;

private:
    // Map field names to data that was read. There are several copies of
    // repeated fields of non primitive types.
    std::map<std::string, ProtobufFields> fields;
    // Map fields tags to their names.
    std::map<int, std::string> nameByTag;
};

}  // namespace pb
}  // namespace cv

#endif  // __OPENCV_PROTOBUF_PARSER_PROTO_TERMS_HPP__
