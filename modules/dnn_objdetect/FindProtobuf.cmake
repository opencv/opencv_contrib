# Protobuf package for Object Detection using CNNs
list(APPEND Protobuf_DIR "/usr/local")
unset(PROTOBUF_FOUND)

find_library(Protobuf_LIBRARY
  NAMES protobuf
  PATH_SUFFIXES build/lib lib
  PATHS "${Protobuf_DIR}"
  DOC "Directory where `protobuf` library is found")

if(Protobuf_LIBRARY)
	set(PROTOBUF_FOUND 1)
	set(PROTOBUF_LIBRARIES ${Protobuf_LIBRARY})
endif(Protobuf_LIBRARY)