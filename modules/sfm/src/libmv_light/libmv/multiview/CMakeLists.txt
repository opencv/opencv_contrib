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
IF(WIN32 AND MSVC)
    string(COMPARE GREATER ${EIGEN_WORLD_VERSION} "2" w3)
    string(COMPARE GREATER ${EIGEN_MAJOR_VERSION} "2" v3)
    IF (w3 AND v3)
         set_target_properties(multiview PROPERTIES  COMPILE_DEFINITIONS "_USE_MATH_DEFINES")
    ENDIF(w3 AND v3)
ENDIF(WIN32 AND MSVC)

LIBMV_INSTALL_LIB(multiview)
