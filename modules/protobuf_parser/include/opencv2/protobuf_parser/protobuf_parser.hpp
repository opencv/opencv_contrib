// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_PROTOBUF_PARSER_PROTOBUF_PARSER_HPP__
#define __OPENCV_PROTOBUF_PARSER_PROTOBUF_PARSER_HPP__

#include <string>
#include <vector>

#include "opencv2/core.hpp"

namespace cv { namespace pb {
//! @addtogroup protobuf_parser
//! @{

    //! Enumeration of protocol buffer fields types supported by parser.
    enum ProtobufFieldType {
      PB_INT32, PB_UINT32, PB_INT64, PB_UINT64,  // Varint
      PB_FLOAT, PB_DOUBLE, PB_BOOL,              // Fixed length
      PB_STRING,                                 // Variable length
      PB_MESSAGE                                 // Variable length, complex types
    };

    /**
     * @brief ProtobufField is an every protobuf entry with type, name and tag.
     *
     * That may be field with elementary type like int32, float, string etc. or
     * more complicated type like enum or message.
     */
    class ProtobufField
    {
    public:
        /**
         * @brief Protobuf field contructor.
         * @param[in] type Type of field. @see ProtobufFieldType
         */
        ProtobufField(int type);

        /**
         * @brief Interpter binary data from stream into field values.
         * @param[in] s Input binary stream.
         */
        virtual void read(std::istream& s) = 0;

        /**
         * @brief Interpter text tokens into field values.
         * @param[in] tokenIt Next iterator to string.
         */
        virtual void read(std::vector<std::string>::iterator& tokenIt) = 0;

        /**
         * @brief Make a copy of field with the same behavior.
         * @return Pointer to new ProtobufField.
         *
         * Copy of behavior means that if new object and the origin one read
         * some data, their values will be the same for the same data and are
         * different in the opposite way. If some of objects reads data then
         * it doesn't affects on values in another one.
         */
        virtual Ptr<ProtobufField> clone() const = 0;

        /**
         * @brief Remove all read data but keep internal state to be ready read it again.
         */
        virtual void clear() = 0;

        virtual bool empty() const = 0;

        /**
         * @brief Returns type of field
         */
        int type() const;

        /**
         * @brief Virtual desctuctor.
         */
        virtual ~ProtobufField() {}

    protected:
        int _type;
    };

    //! Shortcut to vector of protobuf fields.
    typedef std::vector<Ptr<ProtobufField> > ProtobufFields;

    /**
     * @brief Class that is used for access to data from parsed `.pb`.
     */
    class CV_EXPORTS ProtobufNode
    {
    public:
        /**
         * @brief Construct node from a set of fields.
         */
        explicit ProtobufNode(const ProtobufFields& fields = ProtobufFields());

        explicit ProtobufNode(const Ptr<ProtobufField>& field);

        /**
         * @brief Access to embedded node by name.
         * @param[in] name Embedded node name.
         * @return New ProtobufNode that contains certain field.
         */
        ProtobufNode operator[](const std::string& name) const;

        /**
         * @brief Access to embedded node by name.
         * @param[in] name Embedded node name.
         * @return New ProtobufNode that contains certain field.
         */
        ProtobufNode operator[](const char* name) const;

        /**
         * @brief Access to embedded node by index.
         * @param[in] idx Index of element.
         * @return New ProtobufNode that contains certain field.
         *
         * If a field is a value with repeated option, it iterate over values.
         */
        ProtobufNode operator[](int idx) const;

        void operator >> (int32_t& value) const; //!< Retrieve value from node.
        void operator >> (uint32_t& value) const; //!< Retrieve value from node.
        void operator >> (int64_t& value) const; //!< Retrieve value from node.
        void operator >> (uint64_t& value) const; //!< Retrieve value from node.
        void operator >> (float& value) const; //!< Retrieve value from node.
        void operator >> (double& value) const; //!< Retrieve value from node.
        void operator >> (bool& value) const; //!< Retrieve value from node.
        void operator >> (std::string& str) const; //!< Retrieve value from node.

        operator int32_t() const; //!< Explicit cast to retrieve value from node.
        operator int64_t() const; //!< Explicit cast to retrieve value from node.
        operator uint32_t() const; //!< Explicit cast to retrieve value from node.
        operator uint64_t() const; //!< Explicit cast to retrieve value from node.
        operator float() const; //!< Explicit cast to retrieve value from node.
        operator double() const; //!< Explicit cast to retrieve value from node.
        operator bool() const; //!< Explicit cast to retrieve value from node.
        operator std::string() const; //!< Explicit cast to retrieve value from node.

        void set(const std::string& str, int idx = 0); //!< Modify a value inside node.

        int type() const;  //!< Get type of node.

        /**
         * @brief Copy certain number of bytes into pre-allocated memory.
         * @param[in] numBytes Quantity of bytes to read.
         * @param[out] dst Pointer to memory.
         *
         * Use this method in case of big size arrays for better efficiency.
         */
        void copyTo(int numBytes, void* dst) const;

        /**
         * @brief Check that node has no fields.
         *
         * Equals to `node.size() == 0`.
         */
        bool empty() const;

        /**
         * @brief Check that node with specific name was read.
         * @param[in] name Node name.
         */
        bool has(const std::string& name) const;

        //! Returns number of fields or number of repeated values.
        size_t size() const;

        /**
         * @brief Remove field from message.
         * @param[in] name Name of message.
         * @param[in] idx Index of field to remove.
         */
        void remove(const std::string& name, int idx = 0);

        std::vector<std::string> readFields() const;

    private:
        ProtobufFields fields;
    };

    /**
     * @brief Class that parses compiled binary `.pb` and text `.pbtxt` files.
     */
    class CV_EXPORTS ProtobufParser
    {
    public:
        /**
         * @brief Constructor of ProtobufParser.
         * @param[in] bytes Pointer to beginning of compiled `.proto` file.
         * @param[in] numBytes Number of bytes in passed array.
         * @param[in] msg Name of root message for the future parsing.
         * @param[in] text Indicates that compiled `.proto` is represented in text format.
         *
         * Use protocol buffer compiler to make a self-describing message from
         * `.proto` configuration file:
         * ```
         * cd /location/of/example.proto
         * protoc --descriptor_set_out=./example.pb example.proto
         * ```
         *
         * To make a text representation, parse the compiled binary file and save
         * printed text:
         * ```
         * from google.protobuf import descriptor_pb2
         *
         * msg = descriptor_pb2.FileDescriptorSet()
         *
         * with open('/path/to/example.pb', 'rb') as f:
         *     msg.ParseFromString(f.read())
         *     print msg
         * ```
         */
        ProtobufParser(char* bytes, int numBytes, const std::string& msg, bool text = false);

        /**
         * @brief This is an overloaded member function, provided for convenience. It differs from the above function only in what argument(s) it accepts.
         * @param[in] filePath Path to compiled `.proto` file.
         * @param[in] msg Name of root message for the future parsing.
         * @param[in] text Indicates that compiled `.proto` is represented in text format.
         */
        ProtobufParser(const std::string& filePath, const std::string& msg, bool text = false);

        /**
        * @brief This is an overloaded member function, provided for convenience. It differs from the above function only in what argument(s) it accepts.
         * @param[in] s Input stream.
         * @param[in] msg Name of root message for the future parsing.
         * @param[in] text Indicates that compiled `.proto` is represented in text format.
         */
        ProtobufParser(std::istream& s, const std::string& msg, bool text = false);


        /**
         * @brief Parse serialized message.
         * @param[in] bytes Pointer to beginning of serialized data.
         * @param[in] numBytes Number of bytes in passed array.
         * @param[in] text Flag to indicate that file is textual.
         */
        void parse(char* bytes, int numBytes, bool text = false);

        /**
         * @brief This is an overloaded member function, provided for convenience. It differs from the above function only in what argument(s) it accepts.
         * @param[in] filePath Path to file.
         * @param[in] text Flag to indicate that file is textual.
         */
        void parse(const std::string& filePath, bool text = false);

        /**
         * @brief This is an overloaded member function, provided for convenience. It differs from the above function only in what argument(s) it accepts.
         * @param[in] s Input stream.
         * @param[in] text Flag to indicate that data is textual.
         */
        void parse(std::istream& s, bool text = false);

        /**
         * @brief Access to embedded node by name.
         * @param[in] name Embedded node name.
         * @return New ProtobufNode that contains certain field.
         */
        ProtobufNode operator[](const std::string& name) const;

        /**
         * @brief Check that node with specific name was read.
         * @param[in] name Node name.
         */
        bool has(const std::string& name) const;

        /**
         * @brief Remove field from message.
         * @param[in] name Name of message.
         * @param[in] idx Index of field to remove.
         */
        void remove(const std::string& name, int idx = 0);

        //! Returns top node of message as ProtobufNode.
        ProtobufNode root() const;

    private:
        void init(std::istream& s, const std::string& msg, bool text);

        void parseIntoMsg(std::istream& s, bool text);

        Ptr<ProtobufField> message;
    };
//! @}
}  // namespace pb
}  // namespace cv

#endif  // __OPENCV_PROTOBUF_PARSER_PROTOBUF_PARSER_HPP__
