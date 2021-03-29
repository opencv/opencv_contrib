set(the_description "Super Resolution using CNNs")

ocv_define_module(
    dnn_superres
    opencv_core
    opencv_imgproc
    opencv_dnn
    OPTIONAL opencv_quality
    WRAP python
)