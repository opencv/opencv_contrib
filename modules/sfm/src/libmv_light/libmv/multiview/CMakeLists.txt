# define the source files
SET(MULTIVIEW_SRC autocalibration.cc
                  bundle.cc
                  conditioning.cc
                  euclidean_resection.cc
                  fundamental.cc
                  fundamental_kernel.cc
                  projection.cc
                  robust_estimation.cc
                  robust_fundamental.cc
                  robust_resection.cc
                  structure.cc
                  triangulation.cc
                  twoviewtriangulation.cc)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB MULTIVIEW_HDRS *.h)

ADD_LIBRARY(multiview STATIC ${MULTIVIEW_SRC} ${MULTIVIEW_HDRS})
TARGET_LINK_LIBRARIES(multiview glog numeric V3D)

ADD_LIBRARY(multiview_test_data STATIC test_data_sets.cc)
TARGET_LINK_LIBRARIES(multiview_test_data multiview)

LIBMV_INSTALL_LIB(multiview)
LIBMV_INSTALL_LIB(multiview_test_data)