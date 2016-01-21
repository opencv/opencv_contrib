# define the source files
SET(NUMERIC_SRC numeric.cc
                poly.cc)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB NUMERIC_HDRS *.h)

ADD_LIBRARY(numeric STATIC ${NUMERIC_SRC} ${NUMERIC_HDRS})

TARGET_LINK_LIBRARIES(numeric)

LIBMV_INSTALL_LIB(numeric)