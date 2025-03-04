set(the_description "Color Correction Module")
ocv_define_module(ccm
    opencv_core
    opencv_imgproc
    opencv_3d
    opencv_dnn
    WRAP
      python
)
