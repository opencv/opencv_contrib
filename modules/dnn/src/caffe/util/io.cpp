#ifdef HAVE_PROTOBUF
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core.hpp>
//#include <stdint.h>
//
//#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
//#include <string>
//#include <vector>
//
//#include "caffe/common.hpp"
//#include "caffe.pb.h"
//#include "caffe/util/io.hpp"
//
const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::IstreamInputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
    std::ifstream fs(filename, std::ifstream::in);
    CV_Assert(fs.is_open());
    IstreamInputStream input(&fs);
    bool success = google::protobuf::TextFormat::Parse(&input, proto);
    fs.close();
    return success;
}

//
//void WriteProtoToTextFile(const Message& proto, const char* filename) {
//  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
//  FileOutputStream* output = new FileOutputStream(fd);
//  CHECK(google::protobuf::TextFormat::Print(proto, output));
//  delete output;
//  close(fd);
//}
//
bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
    std::ifstream fs(filename, std::ifstream::in | std::ifstream::binary);
    CV_Assert(fs.is_open());
    ZeroCopyInputStream* raw_input = new IstreamInputStream(&fs);
    CodedInputStream* coded_input = new CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

    bool success = proto->ParseFromCodedStream(coded_input);

    delete coded_input;
    delete raw_input;
    fs.close();
    return success;
}
//
//void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
//  fstream output(filename, ios::out | ios::trunc | ios::binary);
//  CHECK(proto.SerializeToOstream(&output));
//}

}  // namespace caffe
#endif