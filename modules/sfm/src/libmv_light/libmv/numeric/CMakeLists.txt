# define the source files
SET(NUMERIC_SRC numeric.cc
                poly.cc)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB NUMERIC_HDRS *.h)

ADD_LIBRARY(numeric STATIC ${NUMERIC_SRC} ${NUMERIC_HDRS})

IF(TARGET Eigen3::Eigen)
  TARGET_LINK_LIBRARIES(numeric LINK_PUBLIC Eigen3::Eigen)
ENDIF()

LIBMV_INSTALL_LIB(numeric)
