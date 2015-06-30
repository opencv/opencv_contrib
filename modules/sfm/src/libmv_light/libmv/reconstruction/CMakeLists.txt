# define the source files
SET(RECONSTRUCTION_SRC mapping.cc
                       optimization.cc
                       projective_reconstruction.cc
                       reconstruction.cc
                       tools.cc)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB RECONSTRUCTION_HDRS *.h)

ADD_LIBRARY(reconstruction STATIC ${RECONSTRUCTION_SRC} ${RECONSTRUCTION_HDRS})

TARGET_LINK_LIBRARIES(reconstruction camera correspondence multiview numeric V3D colamd ldl glog)

LIBMV_INSTALL_LIB(reconstruction)