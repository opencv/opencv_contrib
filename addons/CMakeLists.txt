cmake_minimum_required(VERSION 2.8.12)

set(OpenCV_PACKAGE_NAME OpenCV-Contrib)

set(OpenCV_SKIP_ADDONS ALL) # Import only builtin OpenCV modules
# set(OpenCV_SKIP_ADDONS ${OpenCV_SKIP_ADDONS} ${PACKAGE_NAME})

# Addons support is available in 3.1.x+
find_package(OpenCV 3 REQUIRED)

# Bootstrap addons build helpers
include("${OpenCV_CONFIG_PATH}/addons/cmake/OpenCVAddons.cmake")

macro(addon_module name)
  add_subdirectory(../modules/${name} ${CMAKE_BINARY_DIR}/modules/${name})
endmacro()

addon_module(aruco)
addon_module(bgsegm)
addon_module(bioinspired)
addon_module(ccalib)
addon_module(cnn_3dobj)
#addon_module(contrib_world)
addon_module(cvv)
addon_module(datasets)
#addon_module(dnn)
addon_module(dpm)
addon_module(face)
addon_module(fuzzy)
addon_module(hdf)
addon_module(line_descriptor)
#addon_module(matlab)
addon_module(optflow)
addon_module(plot)
addon_module(reg)
addon_module(rgbd)
addon_module(saliency)
addon_module(sfm)
addon_module(stereo)
addon_module(structured_light)
addon_module(surface_matching)
#addon_module(text)
addon_module(tracking)
addon_module(xfeatures2d)
addon_module(ximgproc)
addon_module(xobjdetect)
addon_module(xphoto)

ocv_addon_finalize()
