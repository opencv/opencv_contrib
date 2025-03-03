set(the_description "Color Correction Module")
ocv_define_module(ccm
    opencv_core
    opencv_imgproc
    opencv_3d
    opencv_dnn
    opencv_mcc
    WRAP
      python
)
add_dependencies(opencv_ccm opencv_mcc)
include_directories(${OPENCV_EXTRA_MODULES_PATH}/modules/mcc/include)
