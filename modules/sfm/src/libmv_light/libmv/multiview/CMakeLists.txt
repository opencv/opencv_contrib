# define the source files
SET(MULTIVIEW_SRC conditioning.cc
                  euclidean_resection.cc
                  fundamental.cc
                  fundamental_kernel.cc
                  homography.cc
                  panography.cc
                  panography_kernel.cc
                  projection.cc
                  robust_estimation.cc
                  robust_fundamental.cc
                  robust_resection.cc
                  triangulation.cc
                  twoviewtriangulation.cc)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB MULTIVIEW_HDRS *.h)

ADD_LIBRARY(multiview STATIC ${MULTIVIEW_SRC} ${MULTIVIEW_HDRS})
TARGET_LINK_LIBRARIES(multiview ${GLOG_LIBRARY} numeric)

LIBMV_INSTALL_LIB(multiview)
