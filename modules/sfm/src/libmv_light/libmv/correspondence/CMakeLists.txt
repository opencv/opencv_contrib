# define the source files
SET(CORRESPONDENCE_SRC feature_matching.cc
                       matches.cc
                       nRobustViewMatching.cc)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB CORRESPONDENCE_HDRS *.h)

ADD_LIBRARY(correspondence STATIC ${CORRESPONDENCE_SRC} ${CORRESPONDENCE_HDRS})

TARGET_LINK_LIBRARIES(correspondence LINK_PRIVATE ${GLOG_LIBRARIES} multiview)
IF(TARGET Eigen3::Eigen)
  TARGET_LINK_LIBRARIES(correspondence LINK_PUBLIC Eigen3::Eigen)
ENDIF()


LIBMV_INSTALL_LIB(correspondence)
