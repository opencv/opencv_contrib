// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "proto_descriptors.hpp"

#include <string>

namespace cv { namespace pb {

    // Descriptor of field options.
    struct FieldOptionsDescriptor : public ProtoMessage
    {
        FieldOptionsDescriptor()
        {
            addField(PB_BOOL, "packed", 2);
            // Unused. Only for text descriptors.
            addField(PB_BOOL, "deprecated", 3);
        }

        static Ptr<ProtobufField> create()
        {
            return Ptr<ProtobufField>(new FieldOptionsDescriptor());
        }
    };

    // Descriptor of field definitions.
    struct FieldDescriptor : public ProtoMessage
    {
        FieldDescriptor()
        {
            addField(PB_STRING, "name", 1);
            addField(PB_INT32, "number", 3);

            Ptr<ProtoEnum> labelEnum(new ProtoEnum(false, ""));
            labelEnum->addValue("LABEL_OPTIONAL", 1);
            labelEnum->addValue("LABEL_REQUIRED", 2);
            labelEnum->addValue("LABEL_REPEATED", 3);
            addField(labelEnum, "label", 4);

            Ptr<ProtoEnum> typeEnum(new ProtoEnum(false, ""));
            typeEnum->addValue("TYPE_DOUBLE", 1);
            typeEnum->addValue("TYPE_FLOAT", 2);
            typeEnum->addValue("TYPE_INT64", 3);
            typeEnum->addValue("TYPE_UINT64", 4);
            typeEnum->addValue("TYPE_INT32", 5);
            typeEnum->addValue("TYPE_BOOL", 8);
            typeEnum->addValue("TYPE_STRING", 9);
            typeEnum->addValue("TYPE_MESSAGE", 11);
            typeEnum->addValue("TYPE_BYTES", 12);
            typeEnum->addValue("TYPE_UINT32", 13);
            typeEnum->addValue("TYPE_ENUM", 14);
            addField(typeEnum, "type", 5);

            addField(PB_STRING, "type_name", 6);
            addField(PB_STRING, "default_value", 7);
            addField(FieldOptionsDescriptor::create(), "options", 8);

            // The fields that will be ignored but should be parsed from text
            // representation of descriptors.
            addField(PB_STRING, "oneof_index", 9);
            addField(PB_STRING, "json_name", 10);
        }

        static Ptr<ProtobufField> create()
        {
            return Ptr<ProtobufField>(new FieldDescriptor());
        }
    };

    // Single enum value. Pair <name, number>.
    struct EnumValueDescriptor : public ProtoMessage
    {
        EnumValueDescriptor()
        {
            addField(PB_STRING, "name", 1);
            addField(PB_INT32, "number", 2);
        }

        static Ptr<ProtobufField> create()
        {
            return Ptr<ProtobufField>(new EnumValueDescriptor());
        }
    };

    // Descriptor of enum definitions.
    struct EnumDescriptor : public ProtoMessage
    {
        EnumDescriptor()
        {
            addField(PB_STRING, "name", 1);
            addField(EnumValueDescriptor::create(), "value", 2);
        }

        static Ptr<ProtobufField> create()
        {
            return Ptr<ProtobufField>(new EnumDescriptor());
        }
    };

    struct OneofDescriptor : public ProtoMessage
    {
        OneofDescriptor()
        {
            addField(PB_STRING, "name", 1);
        }

        static Ptr<ProtobufField> create()
        {
            return Ptr<ProtobufField>(new OneofDescriptor());
        }
    };

    struct MessageOptions : public ProtoMessage
    {
        MessageOptions()
        {
            addField(PB_BOOL, "map_entry", 7);
        }

        static Ptr<ProtobufField> create()
        {
            return Ptr<ProtobufField>(new MessageOptions());
        }
    };

    // Descriptor of message definitions.
    struct MessageDescriptor : public ProtoMessage
    {
        explicit MessageDescriptor(int maxMsgDepth)
        {
            addField(PB_STRING, "name", 1);
            addField(FieldDescriptor::create(), "field", 2);
            if (maxMsgDepth)
            {
                maxMsgDepth -= 1;
                addField(MessageDescriptor::create(maxMsgDepth), "nested_type", 3);
            }
            addField(EnumDescriptor::create(), "enum_type", 4);

            // Unused. Only for text descriptors.
            addField(MessageOptions::create(), "options", 7);
            addField(OneofDescriptor::create(), "oneof_decl", 8);
        }

        static Ptr<ProtobufField> create(int maxMsgDepth)
        {
            return Ptr<ProtobufField>(new MessageDescriptor(maxMsgDepth));
        }
    };

    struct FileOptions : public ProtoMessage
    {
        explicit FileOptions()
        {
            addField(PB_STRING, "java_package", 1);
            addField(PB_STRING, "java_outer_classname", 8);
            addField(PB_BOOL, "java_multiple_files", 10);
            addField(PB_BOOL, "cc_enable_arenas", 31);
        }

        static Ptr<ProtobufField> create()
        {
            return Ptr<ProtobufField>(new FileOptions());
        }
    };

    // Definition of single `.proto` file.
    struct FileDescriptor : public ProtoMessage
    {
        explicit FileDescriptor(int maxMsgDepth)
        {
            addField(PB_STRING, "name", 1);
            addField(PB_STRING, "package", 2);
            addField(PB_STRING, "syntax", 12);
            addField(MessageDescriptor::create(maxMsgDepth), "message_type", 4);
            addField(EnumDescriptor::create(), "enum_type", 5);

            // Unused. Only for text descriptors.
            addField(PB_STRING, "dependency", 3);
            addField(FileOptions::create(), "options", 8);
        }

        static Ptr<ProtobufField> create(int maxMsgDepth)
        {
            return Ptr<ProtobufField>(new FileDescriptor(maxMsgDepth));
        }
    };

    FileDescriptorSet::FileDescriptorSet(int maxMsgDepth)
    {
        addField(FileDescriptor::create(maxMsgDepth), "file", 1);
    }

}  // namespace pb
}  // namespace cv
