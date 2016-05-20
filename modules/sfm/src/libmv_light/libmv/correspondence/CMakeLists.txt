# define the source files
SET(CORRESPONDENCE_SRC feature_matching.cc
                       matches.cc
                       nRobustViewMatching.cc)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB CORRESPONDENCE_HDRS *.h)

ADD_LIBRARY(correspondence STATIC ${CORRESPONDENCE_SRC} ${CORRESPONDENCE_HDRS})

TARGET_LINK_LIBRARIES(correspondence multiview)

LIBMV_INSTALL_LIB(correspondence)