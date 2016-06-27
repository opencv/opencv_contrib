#Required for Caffe
unset(Glog_FOUND)

find_library(Glog_LIBS NAMES glog
  HINTS
  /usr/local/lib)

if(Glog_LIBS)
    set(Glog_FOUND 1)
endif()
