// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace cvtest
{
    using namespace cv;
    using namespace cv::pb;

    static int toInt(const ProtobufNode& node)
    {
        switch (node.type())
        {
            case PB_INT32: return (int32_t)node;
            case PB_UINT32: return (int)(uint32_t)node;
            case PB_INT64: return (int)(int64_t)node;
            case PB_UINT64: return (int)(uint64_t)node;
            case PB_BOOL: return (bool)node;
            default:
                CV_Error(Error::StsUnsupportedFormat, "Type mismatch");
        }
        return 0;
    }

    static float toReal(const ProtobufNode& node)
    {
        switch (node.type())
        {
            case PB_FLOAT: return (float)node;
            case PB_DOUBLE: return (float)(double)node;
            default:
                CV_Error(Error::StsUnsupportedFormat, "Type mismatch");
        }
        return 0;
    }

    static void testValue(const FileNode& jsNode, const ProtobufNode& pbNode)
    {
        if (jsNode.isInt())
        {
            ASSERT_EQ(toInt(pbNode), (int)jsNode);
        }
        else if (jsNode.isReal())
        {
            ASSERT_EQ(toReal(pbNode), (float)jsNode);
        }
    }

    static void test(const FileNode& jsNode, const ProtobufNode& pbNode)
    {
        for (FileNodeIterator it = jsNode.begin(); it != jsNode.end(); ++it)
        {
            const FileNode& field = *it;
            ASSERT_TRUE(field.isNamed());
            if (field.size())
            {
                ASSERT_TRUE(pbNode.has(field.name()));
                const ProtobufNode& node = pbNode[field.name()];
                if (field.isMap())
                {
                    test(field, node);
                }
                else if (field.isSeq())
                {
                    for (int i = 0; i < (int)field.size(); ++i)
                        testValue(field[i], node[i]);
                }
                else if (field.isString())
                {
                    ASSERT_EQ((std::string)node, (std::string)field);
                }
                else
                {
                    testValue(field, node);
                }
            }
            else
            {
                ASSERT_TRUE(field.isNamed());
                ASSERT_FALSE(pbNode.has(field.name()));
            }
        }
    }

    static ProtobufParser test(const std::string& prefix, const std::string& msg,
                               std::string proto)
    {
        proto = "protobuf_parser/bin/" + proto;
        ProtobufParser pp(findDataFile(proto, false), msg);
        pp.parse(findDataFile("protobuf_parser/bin/" + prefix + ".pb", false));

        std::string json = findDataFile("protobuf_parser/test_data.json", false);
        FileStorage fs(json, FileStorage::READ);
        test(fs[prefix], pp.root());
        fs.release();

        return pp;
    }

    TEST(ProtobufParser, SimpleValues)
    {
        test("simple_values", ".SimpleValues", "test.pb");
    }

    TEST(ProtobufParser, NestedMessage)
    {
        ProtobufParser pp = test("nested_message", ".HostMsg", "test.pb");
        ASSERT_EQ((float)pp["nested_msg_value"][0]["value_float_default"], -0.9f);
        ASSERT_EQ((float)pp["nested_msg_value"][1]["value_float_default"], -0.9f);
    }

    TEST(ProtobufParser, DefaultValues)
    {
        ProtobufParser pp = test("default_values", ".DefaultValues", "test.pb");
        ASSERT_EQ((bool)pp["value_true"], true);
        ASSERT_EQ((bool)pp["value_false"], false);
        ASSERT_EQ((double)pp["value_double"], 1e-2);
    }

    TEST(ProtobufParser, Enums)
    {
        ProtobufParser pp = test("enums", ".EnumValues", "test.pb");
        ASSERT_EQ((std::string)pp["enum_value_default"], "MACRO_ENUM_VALUE_3");
    }

    TEST(ProtobufParser, PackedValues)
    {
        ProtobufParser pp = test("packed_values", ".PackedValues", "test.pb");
        ASSERT_EQ((float)pp["nested_msg"][0]["value_float_default"], -0.9f);
        ASSERT_EQ((float)pp["nested_msg"][1]["value_float_default"], -0.9f);
    }

    TEST(ProtobufParser, Package)
    {
        test("package", ".test.MessageTwo", "test_package.pb");
    }

    TEST(ProtobufParser, Map)
    {
        const std::string proto = "protobuf_parser/bin/test_proto3.pb";
        ProtobufParser pp(findDataFile(proto, false), ".Map");
        pp.parse(findDataFile("protobuf_parser/bin/map.pb", false));

        ASSERT_TRUE(pp.has("int_to_string"));
        ASSERT_TRUE(pp.has("string_to_float"));
        ASSERT_TRUE(pp.has("string_to_mix"));

        ASSERT_EQ(pp["int_to_string"].size(), (size_t)2);
        ASSERT_EQ(pp["string_to_float"].size(), (size_t)2);
        ASSERT_EQ(pp["string_to_mix"].size(), (size_t)2);

        ASSERT_EQ((int)pp["int_to_string"][0]["key"], 1);
        ASSERT_EQ((int)pp["int_to_string"][1]["key"], -2);
        ASSERT_EQ((std::string)pp["int_to_string"][0]["value"], "first string");
        ASSERT_EQ((std::string)pp["int_to_string"][1]["value"], "string with negative key");

        ASSERT_EQ((std::string)pp["string_to_float"][0]["key"], "key2");
        ASSERT_EQ((std::string)pp["string_to_float"][1]["key"], "key1");
        ASSERT_EQ((float)pp["string_to_float"][0]["value"], 8.05f);
        ASSERT_EQ((float)pp["string_to_float"][1]["value"], -9.321f);

        ASSERT_EQ((std::string)pp["string_to_mix"][0]["key"], "efgh");
        ASSERT_EQ((uint64_t)pp["string_to_mix"][0]["value"]["value"][0], 42ul);

        ASSERT_EQ((std::string)pp["string_to_mix"][1]["key"], "abcd");
        ASSERT_EQ((uint64_t)pp["string_to_mix"][1]["value"]["value"][0], 124ul);
        ASSERT_EQ((uint64_t)pp["string_to_mix"][1]["value"]["value"][1], 12ul);
        ASSERT_EQ((uint64_t)pp["string_to_mix"][1]["value"]["value"][2], 0ul);
        ASSERT_EQ((std::string)pp["string_to_mix"][1]["value"]["str"], "mystr");
    }
}
