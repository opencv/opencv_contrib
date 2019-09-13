if(HAVE_CUDA)
  add_definitions(-D_HFS_CUDA_ON_)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
endif()

set(the_description "Hierarchical Feature Selection for Efficient Image Segmentation")
ocv_define_module(hfs opencv_core opencv_imgproc WRAP python)