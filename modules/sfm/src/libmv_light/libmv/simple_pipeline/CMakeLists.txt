SET(SIMPLE_PIPELINE_SRC
    bundle.cc
    camera_intrinsics.cc
    initialize_reconstruction.cc
    intersect.cc
    pipeline.cc
    reconstruction.cc
    resect.cc
    tracks.cc)

# Define the header files so that they appear in IDEs.
FILE(GLOB SIMPLE_PIPELINE_HDRS *.h)

ADD_LIBRARY(simple_pipeline STATIC ${SIMPLE_PIPELINE_SRC} ${SIMPLE_PIPELINE_HDRS})

TARGET_LINK_LIBRARIES(simple_pipeline V3D multiview ceres)

LIBMV_INSTALL_LIB(simple_pipeline)