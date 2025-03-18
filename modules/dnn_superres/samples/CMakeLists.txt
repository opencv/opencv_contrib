set(OPENCV_SUPERRES_SAMPLE_REQUIRED_DEPS
    opencv_core
    opencv_imgproc
    opencv_dnn_superres
    opencv_highgui
    opencv_imgcodecs
)

ocv_add_executable(dnn_superres
    dnn_superres.cpp
)

ocv_add_executable(dnn_superres_benchmark
    dnn_superres_benchmark.cpp
)

ocv_add_executable(dnn_superres_srgan_rdn_demo
    dnn_superres_srgan_rdn_demo.cpp
)

ocv_target_link_libraries(dnn_superres
    ${OPENCV_SUPERRES_SAMPLE_REQUIRED_DEPS}
)

ocv_target_link_libraries(dnn_superres_benchmark
    ${OPENCV_SUPERRES_SAMPLE_REQUIRED_DEPS}
)

ocv_target_link_libraries(dnn_superres_srgan_rdn_demo
    ${OPENCV_SUPERRES_SAMPLE_REQUIRED_DEPS}
) 
