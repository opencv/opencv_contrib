# define the source files
SET(CAMERA_SRC camera.cc
               lens_distortion.cc
               pinhole_camera.cc)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB CAMERA_HDRS *.h)

ADD_LIBRARY(camera STATIC ${CAMERA_SRC} ${CAMERA_HDRS})

TARGET_LINK_LIBRARIES(camera multiview numeric V3D colamd ldl)

LIBMV_INSTALL_LIB(camera)