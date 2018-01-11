# define the source files
SET(BASE_SRC )

# define the header files (make the headers appear in IDEs.)
FILE(GLOB BASE_HDRS *.h)

ADD_LIBRARY(base STATIC ${BASE_SRC} ${BASE_HDRS})

LIBMV_INSTALL_LIB(base)