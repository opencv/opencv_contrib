set(the_description "Tracking API")

set(debug_modules "")
if(DEBUG_opencv_tracking)
  list(APPEND debug_modules opencv_highgui)
endif()

ocv_define_module(tracking
    opencv_imgproc
    opencv_core
    opencv_video
    opencv_plot  # samples only
    ${debug_modules}
    OPTIONAL
      opencv_dnn
      opencv_datasets
      opencv_highgui
    WRAP
      java
      python
      objc
)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-shadow /wd4458)

if(TARGET opencv_test_${name})
  ocv_target_include_directories(opencv_test_${name} "${OpenCV_SOURCE_DIR}/modules")  # use common files from video tests
endif()
