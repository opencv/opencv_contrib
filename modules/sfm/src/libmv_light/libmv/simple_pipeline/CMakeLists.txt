SET(SIMPLE_PIPELINE_SRC
    bundle.cc
    camera_intrinsics.cc
    distortion_models.cc
    initialize_reconstruction.cc
    intersect.cc
    keyframe_selection.cc
    pipeline.cc
    reconstruction.cc
    reconstruction_scale.cc
    resect.cc
    tracks.cc
)

# Define the header files so that they appear in IDEs.
FILE(GLOB SIMPLE_PIPELINE_HDRS *.h)

ADD_LIBRARY(opencv.sfm.simple_pipeline STATIC ${SIMPLE_PIPELINE_SRC} ${SIMPLE_PIPELINE_HDRS})

TARGET_LINK_LIBRARIES(opencv.sfm.simple_pipeline LINK_PRIVATE opencv.sfm.multiview ${CERES_LIBRARIES})

LIBMV_INSTALL_LIB(opencv.sfm.simple_pipeline)
