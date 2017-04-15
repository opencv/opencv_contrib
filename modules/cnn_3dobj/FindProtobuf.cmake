# Protobuf package for CNN Triplet training
unset(Protobuf_FOUND)

find_library(Protobuf_LIBS NAMES protobuf
  HINTS
  /usr/local/lib)

if(Protobuf_LIBS)
    set(Protobuf_FOUND 1)
endif()
