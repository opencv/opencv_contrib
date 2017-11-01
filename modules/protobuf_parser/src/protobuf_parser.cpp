// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <utility>
#include <algorithm>
#include "proto_descriptors.hpp"
#include "proto_terms.hpp"

namespace cv { namespace pb {

ProtobufField::ProtobufField(int typeId) : _type(typeId) {}

int ProtobufField::type() const
{
    return _type;
}

// Remove comments from prototxt file. Let comment is a sequence of characters
// that starts from '#' (inclusive) and ends by '\n' (inclusive).
static std::string removeProtoComments(const std::string& str)
{
    std::string res = "";
    bool isComment = false;
    for (size_t i = 0, n = str.size(); i < n; ++i)
    {
        if (str[i] == '#')
        {
            isComment = true;
        }
        else
        {
            if (isComment)
            {
                isComment = str[i] != '\n';
            }
            else
            {
                res += str[i];
            }
        }
    }
    return res;
}

// Split source text by tokens.
// Delimeters are specific for protobuf in text format.
static std::vector<std::string> tokenize(const std::string& str)
{
    std::vector<std::string> tokens;
    tokens.reserve(max(1, (int)str.size() / 7));

    std::string token = "";
    bool isString = false;  // Flag to manage empty strings.
    for (size_t i = 0, n = str.size(); i < n; ++i)
    {
        char symbol = str[i];
        if (symbol == ' ' || symbol == '\t' || symbol == '\r' ||
            symbol == '\n' || symbol == ':' || symbol == '\"' || symbol == ';')
        {
            if (!token.empty() || (symbol == '\"' && isString))
            {
                tokens.push_back(token);
                token = "";
            }
            isString = (symbol == '\"') ^ isString;
        }
        else if (symbol == '{' || symbol == '}')
        {
            if (!token.empty())
            {
                tokens.push_back(token);
                token = "";
            }
            tokens.push_back(std::string(1, symbol));
        }
        else
        {
            token += symbol;
        }
    }
    if (!token.empty())
    {
        tokens.push_back(token);
    }
    return tokens;
}

// Recursive function for the next procedure.
static void extractTypeNodes(const ProtobufNode& types,
                             const std::string& parentTypeName,
                             std::map<std::string, ProtobufNode>& typeNodes)
{
    std::string typeName;
    ProtobufNode typeNode;
    for (int i = 0, n = (int)types.size(); i < n; ++i)
    {
        typeNode = types[i];

        CV_Assert(typeNode.has("name"));
        typeNode["name"] >> typeName;
        typeName = parentTypeName + "." + typeName;

        std::pair<std::string, ProtobufNode> mapValue(typeName, typeNode);
        CV_Assert(typeNodes.insert(mapValue).second);

        if (typeNode.has("message_type"))
            extractTypeNodes(typeNode["message_type"], typeName, typeNodes);
        if (typeNode.has("nested_type"))
            extractTypeNodes(typeNode["nested_type"], typeName, typeNodes);
        if (typeNode.has("enum_type"))
            extractTypeNodes(typeNode["enum_type"], typeName, typeNodes);
    }
}

// Extract all nodes for combined types -- messages and enums.
// Map them by names.
static void extractTypeNodes(const ProtobufNode& protoRoot,
                             std::map<std::string, ProtobufNode>& typeNodes)
{
    std::string packageName = "";
    if (protoRoot.has("package"))
    {
        protoRoot["package"] >> packageName;
        packageName = "." + packageName;
    }

    if (protoRoot.has("message_type"))
        extractTypeNodes(protoRoot["message_type"], packageName, typeNodes);
    if (protoRoot.has("nested_type"))
        extractTypeNodes(protoRoot["nested_type"], packageName, typeNodes);
    if (protoRoot.has("enum_type"))
        extractTypeNodes(protoRoot["enum_type"], packageName, typeNodes);
}

static Ptr<ProtobufField> buildEnum(const std::string& name,
                                    const std::map<std::string, ProtobufNode>& typeNodes,
                                    const std::string& defaultValue, bool packed)
{
    if (typeNodes.find(name) == typeNodes.end())
        CV_Error(Error::StsParseError, "Enum " + name + " not found");
    const ProtobufNode& enumNode = typeNodes.find(name)->second;

    Ptr<ProtoEnum> enumValue(new ProtoEnum(packed, defaultValue));
    ProtobufNode values = enumNode["value"];
    for (int i = 0; i < (int)values.size(); ++i)
    {
        enumValue->addValue(values[i]["name"], values[i]["number"]);
    }
    return enumValue;
}

static int getProtoType(const std::string& type)
{
    if (type == "TYPE_INT32")       return PB_INT32;
    else if (type == "TYPE_UINT32") return PB_UINT32;
    else if (type == "TYPE_INT64")  return PB_INT64;
    else if (type == "TYPE_UINT64") return PB_UINT64;
    else if (type == "TYPE_FLOAT")  return PB_FLOAT;
    else if (type == "TYPE_DOUBLE") return PB_DOUBLE;
    else if (type == "TYPE_BOOL")   return PB_BOOL;
    else if (type == "TYPE_STRING" || type == "TYPE_BYTES") return PB_STRING;
    else
        CV_Error(Error::StsNotImplemented, "Unknown protobuf type " + type);
    return -1;
}

static Ptr<ProtoMessage> buildMessage(const std::string& name,
                                      const std::map<std::string, ProtobufNode>& typeNodes,
                                      std::map<std::string, Ptr<ProtoMessage> >& builtMessages,
                                      bool proto3)
{
    // Try to find already built message.
    if (builtMessages.find(name) != builtMessages.end())
        return builtMessages[name]->clone().dynamicCast<ProtoMessage>();

    if (typeNodes.find(name) == typeNodes.end())
        CV_Error(Error::StsParseError, "Message name " + name + " not found");
    const ProtobufNode& messageNode = typeNodes.find(name)->second;

    Ptr<ProtoMessage> message(new ProtoMessage());
    builtMessages[name] = message;

    // Get fields.
    ProtobufNode fields = messageNode["field"];
    for (int i = 0; i < (int)fields.size(); ++i)
    {
        ProtobufNode fieldNode = fields[i];

        CV_Assert(fieldNode.has("name"));
        CV_Assert(fieldNode.has("number"));
        CV_Assert(fieldNode.has("type"));
        CV_Assert(fieldNode.has("label"));

        std::string fieldName = fieldNode["name"];
        // typeName is empty string for elementary types or name of enum or message otherwise.
        std::string typeName = fieldNode.has("type_name") ? (std::string)fieldNode["type_name"] : "";
        int fieldTag = fieldNode["number"];
        std::string fieldType = fieldNode["type"];

        // Default value.
        std::string defaultValue = "";
        if (fieldNode.has("default_value"))
        {
            fieldNode["default_value"] >> defaultValue;
        }

        bool packed = (fieldNode.has("options") &&
                      fieldNode["options"].has("packed") &&
                      fieldNode["options"]["packed"]) ||
                      (proto3 && (std::string)fieldNode["label"] == "LABEL_REPEATED");

        Ptr<ProtobufField> field;
        if (fieldType == "TYPE_MESSAGE")
        {
            field = buildMessage(typeName, typeNodes, builtMessages, proto3);
        }
        else if (fieldType == "TYPE_ENUM")
        {
            field = buildEnum(typeName, typeNodes, defaultValue, packed);
        }
        else  // One of the simple types: int32, float, string, etc.
        {
            field = Ptr<ProtobufField>(new ProtoValue(getProtoType(fieldType), packed, defaultValue));
        }
        if (field.empty())
            CV_Error(Error::StsParseError, "Type name " + name + " not found");
        message->addField(field, fieldName, fieldTag);
    }
    return message;
};

ProtobufParser::ProtobufParser(const std::string& filePath, const std::string& msg, bool text)
{
    std::ifstream ifs(filePath.c_str(), text ? std::ios::in : std::ios::binary);
    CV_Assert(ifs.is_open());
    init(ifs, msg, text);
}

ProtobufParser::ProtobufParser(char* bytes, int numBytes, const std::string& msg, bool text)
{
    std::istringstream s(std::string(bytes, numBytes));
    init(s, msg, text);
}

ProtobufParser::ProtobufParser(std::istream& s, const std::string& msg, bool text)
{
    init(s, msg, text);
}

void ProtobufParser::parse(const std::string& filePath, bool text)
{
    std::ifstream ifs(filePath.c_str(), text ? std::ios::in : std::ios::binary);
    CV_Assert(ifs.is_open());
    parseIntoMsg(ifs, text);
}

void ProtobufParser::parse(char* bytes, int numBytes, bool text)
{
    std::istringstream s(std::string(bytes, numBytes));
    parseIntoMsg(s, text);
}

void ProtobufParser::parse(std::istream& s, bool text)
{
    parseIntoMsg(s, text);
}

void ProtobufParser::parseIntoMsg(std::istream& is, bool text)
{
    CV_Assert(!message.empty());
    message->clear();
    if (text)
    {
        is.seekg(0, std::ios::end);
        std::string content((int)is.tellg(), ' ');
        is.seekg(0, std::ios::beg);
        is.read(&content[0], content.size());
        // Add brackets to unify top-level message format. It's easier for text
        // format because in binary format we must write Varint value with
        // top message length.
        content = '{' + content + '}';

        content = removeProtoComments(content);
        std::vector<std::string> tokens = tokenize(content);
        std::vector<std::string>::iterator tokenIt = tokens.begin();
        message->read(tokenIt);
    }
    else
    {
        message->read(is);
    }
}

void ProtobufParser::init(std::istream& s, const std::string& msg, bool text)
{
    message = Ptr<FileDescriptorSet>(new FileDescriptorSet());
    parseIntoMsg(s, text);

    ProtobufNode protoDescriptor(message);

    std::map<std::string, ProtobufNode> typeNodes;
    std::map<std::string, Ptr<ProtoMessage> > builtMessages;
    bool proto3 = false;
    for (int i = 0, n = (int)protoDescriptor["file"].size(); i < n; ++i)
    {
        extractTypeNodes(protoDescriptor["file"][i], typeNodes);
        proto3 = proto3 || (protoDescriptor["file"][i].has("syntax") &&
                 (std::string)protoDescriptor["file"][i]["syntax"] == "proto3");
    }
    message = buildMessage(msg, typeNodes, builtMessages, proto3);
}

ProtobufNode ProtobufParser::operator[](const std::string& name) const
{
    return message.dynamicCast<ProtoMessage>()->operator[](name);
}

bool ProtobufParser::has(const std::string& name) const
{
    return message.dynamicCast<ProtoMessage>()->has(name);
}

void ProtobufParser::remove(const std::string& name, int idx)
{
    message.dynamicCast<ProtoMessage>()->remove(name, idx);
}

ProtobufNode ProtobufParser::root() const
{
    return ProtobufNode(ProtobufFields(1, message));
}

}  // namespace pb
}  // namespace cv
