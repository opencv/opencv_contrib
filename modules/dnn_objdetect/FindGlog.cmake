# Glog package for Object Detection using CNNs
unset(GLOG_FOUND)

find_library(Glog_LIBRARY
  NAMES glog
  HINTS /usr/local/lib /usr/lib
  DOC "Directory where `glog` library is found")

if(Glog_LIBRARY)
	set(GLOG_FOUND 1)
	set(GLOB_LIBRARIES "${Glog_LIBRARY}")
endif(Glog_LIBRARY)