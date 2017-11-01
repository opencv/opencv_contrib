import argparse
from google.protobuf import descriptor_pb2, text_format

parser = argparse.ArgumentParser(
        description='This script creates a C++ header file that contains a single'
                    'std::string object with compiled .proto file in text format.')
parser.add_argument('-i', dest='input', required=True,
                    help='Path to compiled .proto file. Use protobuf compiler: '
                         'protoc --descriptor_set_out=./example.pb example.proto')
parser.add_argument('-o', dest='output', required=True,
                    help='Path to output header file')
parser.add_argument('-d', dest='definition', default='__OPENCV_MODULE_NAME_FILE_NAME_HPP__',
                    help='Definition macros')
parser.add_argument('-v', dest='variable', default='tmp',
                    help='String variable name')
args = parser.parse_args()

msg = descriptor_pb2.FileDescriptorSet()
with open(args.input, 'rb') as f:
    msg.ParseFromString(f.read())
    msg = text_format.MessageToString(msg)

MAX_LITERAL_LENGTH = 60000
literalLength = 0
with open(args.output, 'wt') as f:
    f.write('#ifndef %s\n' % args.definition)
    f.write('#define %s\n' % args.definition)

    msg = msg.replace('\"', '\\"')

    f.write('const std::string %s = std::string("") +\n' % args.variable)
    lines = msg.rstrip('\n').split('\n')
    for line in lines[:-1]:
        literalLength += len(line)
        if literalLength > MAX_LITERAL_LENGTH:
            f.write('+\n')
            literalLength = len(line)
        f.write('\"%s\\n\"\n' % line)
    f.write('\"%s\";\n' % lines[-1])
    f.write('#endif  // %s\n' % args.definition)
