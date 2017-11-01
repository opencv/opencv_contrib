// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

// In this file represented set of protobuf messages that are used for parsing
// compiled `.proto` files. Based on `google/protobuf/descriptor.proto`.

#ifndef __OPENCV_PROTOBUF_PARSER_PROTO_DESCRIPTORS_HPP__
#define __OPENCV_PROTOBUF_PARSER_PROTO_DESCRIPTORS_HPP__

#include <string>

#include "proto_terms.hpp"

namespace cv { namespace pb {

// Set of `.proto` files compiled together (using --include_imports flag of
// proto compiler).
struct FileDescriptorSet : public ProtoMessage
{
    explicit FileDescriptorSet(int maxMsgDepth = 3);
};

}  // namespace pb
}  // namespace cv

#endif  // __OPENCV_PROTOBUF_PARSER_PROTO_DESCRIPTORS_HPP__
