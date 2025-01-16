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

ADD_LIBRARY(opencv.sfm.multiview STATIC ${MULTIVIEW_SRC} ${MULTIVIEW_HDRS})
TARGET_LINK_LIBRARIES(opencv.sfm.multiview LINK_PRIVATE ${GLOG_LIBRARIES} opencv.sfm.numeric)
IF(TARGET Eigen3::Eigen)
  TARGET_LINK_LIBRARIES(opencv.sfm.multiview LINK_PUBLIC Eigen3::Eigen)
ENDIF()
IF(CERES_LIBRARIES)
  TARGET_LINK_LIBRARIES(opencv.sfm.multiview LINK_PRIVATE ${CERES_LIBRARIES})
ENDIF()

LIBMV_INSTALL_LIB(opencv.sfm.multiview)
