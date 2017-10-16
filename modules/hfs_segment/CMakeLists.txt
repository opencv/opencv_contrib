if(NOT HAVE_CUDA)
  ocv_module_disable(hfs_segment)
endif()

set(the_description "Hierarchical Feature Selection for Efficient Image Segmentation")
ocv_define_module(hfs_segment opencv_core opencv_imgproc)