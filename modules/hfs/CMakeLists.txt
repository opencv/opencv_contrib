if(NOT HAVE_CUDA)
  ocv_module_disable(hfs)
endif()

set(the_description "Hierarchical Feature Selection for Efficient Image Segmentation")
ocv_define_module(hfs opencv_core opencv_imgproc WRAP python)