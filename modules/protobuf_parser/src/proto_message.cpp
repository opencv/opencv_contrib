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
#include <iostream>
namespace cv { namespace pb {

// Read varint and extract tag and wire type. Wire type is one of the followings:
// | Wire type |                   Term types |
// |-----------|------------------------------|
// |         0 | int32, int64, uint32, uint64 |
// |           | sint32, sint64, bool, enum   |
// |         1 | fixed64, sfixed64, double    |
// |         2 | string, bytes,               |
// |           | embedded messages,           |
// |           | packed repeated fields       |
// |         5 | fixed32, sfixed32, float     |
// See https://developers.google.com/protocol-buffers/docs/encoding#structure
// Wire types 3 and 4 are deprecated.
static void parseKey(std::istream& s, int* tag, int* wireType)
{
    ProtoValue keyValue(PB_INT32, s);
    if (!s.eof())
    {
        int key = keyValue.getInt32();
        *tag = (key >> 3);
        *wireType = key & 7;  // Last three bits.
        if (*tag <= 0)
        {
            CV_Error(Error::StsParseError, format("Unsupported tag value [%d]", *tag));
        }
        if (*wireType != 0 && *wireType != 1 && *wireType != 2 && *wireType != 5)
        {
            CV_Error(Error::StsParseError, format("Unsupported wire type [%d]", *wireType));
        }
    }
}

ProtoMessage::ProtoMessage() : ProtobufField(PB_MESSAGE) {}

void ProtoMessage::addField(const Ptr<ProtobufField>& field,
                            const std::string& name, int tag)
{
    CV_Assert(nameByTag.find(tag) == nameByTag.end());
    nameByTag[tag] = name;
    fields[name] = ProtobufFields(1, field);
}

void ProtoMessage::addField(int typeId, const std::string& name, int tag)
{
    addField(Ptr<ProtobufField>(new ProtoValue(typeId, false)), name, tag);
}

void ProtoMessage::clear()
{
    std::map<std::string, ProtobufFields>::iterator it;
    for (it = fields.begin(); it != fields.end(); ++it)
    {
        CV_Assert(!it->second.empty());
        it->second.resize(1);
        it->second[0]->clear();
    }
}

void ProtoMessage::read(std::istream& s)
{
    // Start parsing the message.
    int tag = -1, wireType = -1, msgEnd = INT_MAX;
    // Top level message has no length value at the beginning.
    bool isEmbedded = (int)s.tellg() != 0;
    if (isEmbedded)
    {
        // Embedded messages starts from length value.
        int numBytes = ProtoValue(PB_INT32, s).getInt32();
        msgEnd = (int)s.tellg() + numBytes;
    }

    std::map<int, std::string>::iterator nameIt;
    std::map<std::string, ProtobufFields>::iterator fieldIt;
    while (s.tellg() < msgEnd)
    {
        parseKey(s, &tag, &wireType);
        if (s.eof())
        {
            break;
        }

        nameIt = nameByTag.find(tag);
        if (nameIt != nameByTag.end())
        {
            const std::string& fieldName = nameIt->second;

            fieldIt = fields.find(fieldName);
            CV_Assert(fieldIt != fields.end());

            Ptr<ProtobufField> field = fieldIt->second[0];
            switch (field->type())
            {
                case PB_INT32: case PB_UINT32: case PB_INT64: case PB_UINT64:
                case PB_FLOAT: case PB_DOUBLE: case PB_BOOL: case PB_STRING:
                {
                    field->read(s);
                    break;
                }
                case PB_MESSAGE:
                {
                    if (field->empty())
                        field->read(s);
                    else
                    {
                        fieldIt->second.push_back(field->clone());
                        fieldIt->second.back()->read(s);
                    }
                    break;
                }
                default:
                    CV_Error(Error::StsParseError, "Unknown field type");
            }
        }
        else
        {
            // Skip bytes.
            if (wireType == 0)            // Varint.
                ProtoValue(PB_INT64, s);  // Use value with maximal buffer.
            else if (wireType == 1)
                s.ignore(8);  // 64bit value.
            else if (wireType == 2)  // Some set of bytes with length value.
            {
                int numBytes = ProtoValue(PB_INT32, s).getInt32();
                s.ignore(numBytes);
            }
            else if (wireType == 5)
                s.ignore(4);  // 32bit value.
        }
    }
    CV_Assert(!isEmbedded || s.eof() || (int)s.tellg() == msgEnd);
}

void ProtoMessage::read(std::vector<std::string>::iterator& tokenIt)
{
    // Start parsing the message.
    CV_Assert(*tokenIt == "{");
    ++tokenIt;

    std::string fieldName;
    std::map<std::string, ProtobufFields>::iterator fieldIt;
    while (*tokenIt != "}")
    {
        fieldName = *tokenIt;
        ++tokenIt;

        fieldIt = fields.find(fieldName);
        if (fieldIt == fields.end())
            CV_Error(Error::StsNotImplemented, "Unknown field name: " + fieldName);

        // Parse.
        Ptr<ProtobufField> field = fieldIt->second[0];
        switch (field->type())
        {
            case PB_INT32: case PB_UINT32: case PB_INT64: case PB_UINT64:
            case PB_FLOAT: case PB_DOUBLE: case PB_BOOL: case PB_STRING:
            {
                field->read(tokenIt);
                break;
            }
            case PB_MESSAGE:
            {
                if (field->empty())
                    field->read(tokenIt);
                else
                {
                    fieldIt->second.push_back(field->clone());
                    fieldIt->second.back()->read(tokenIt);
                }
                break;
            }
            default:
                CV_Error(Error::StsParseError, "Unknown field type");
        }
    }
    ++tokenIt;
}

Ptr<ProtobufField> ProtoMessage::clone() const
{
    Ptr<ProtoMessage> message(new ProtoMessage());
    message->nameByTag = nameByTag;

    std::map<std::string, ProtobufFields>::const_iterator it;
    for (it = fields.begin(); it != fields.end(); ++it)
    {
        message->fields[it->first] = ProtobufFields(1, it->second[0]->clone());
    }
    return message;
}

ProtobufNode ProtoMessage::operator[](const std::string& name) const
{
    std::map<std::string, ProtobufFields>::const_iterator fieldIt;
    fieldIt = fields.find(name);
    if (fieldIt == fields.end())
    {
        CV_Error(Error::StsObjectNotFound, "There is no field with name " + name);
    }
    return ProtobufNode(fieldIt->second);
}

bool ProtoMessage::has(const std::string& name) const
{
    std::map<std::string, ProtobufFields>::const_iterator fieldIt;
    fieldIt = fields.find(name);

    if (fieldIt == fields.end())
        return false;

    CV_Assert(!fieldIt->second.empty());
    return !fieldIt->second[0]->empty();
}

void ProtoMessage::remove(const std::string& name, int idx)
{
    std::map<std::string, ProtobufFields>::iterator it = fields.find(name);
    CV_Assert(it != fields.end());
    CV_Assert(0 <= idx && idx < (int)it->second.size());
    it->second.erase(it->second.begin() + idx);
}

bool ProtoMessage::empty() const
{
    std::map<std::string, ProtobufFields>::const_iterator it;
    for (it = fields.begin(); it != fields.end(); ++it)
    {
        if (it->second.size() > 1 || !it->second[0]->empty())
            return false;
    }
    return true;
}

std::vector<std::string> ProtoMessage::readFields() const
{
    std::map<std::string, ProtobufFields>::const_iterator it;
    std::vector<std::string> names;
    names.reserve(fields.size());
    for (it = fields.begin(); it != fields.end(); ++it)
    {
        if (!it->second.empty() && !it->second[0].empty())
        {
            names.push_back(it->first);
        }
    }
    return names;
}

}  // namespace pb
}  // namespace cv
