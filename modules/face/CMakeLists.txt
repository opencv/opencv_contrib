set(the_description "Face recognition etc")
ocv_define_module(face opencv_core opencv_imgproc opencv_objdetect WRAP python)
# NOTE: objdetect module is needed for one of the samples
