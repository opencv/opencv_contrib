set(the_description "SLAM module: visual odometry / mapping / localization utilities")

# Core dependencies used by the sources (see includes in src/ and include/)
ocv_define_module(slam
	opencv_core
	opencv_imgproc
	opencv_calib3d
	opencv_features2d
	OPTIONAL opencv_sfm
	opencv_imgcodecs
	opencv_highgui
	opencv_video
	WRAP python
)

# Ensure C++17 is enabled for this module (required for <filesystem>)
if(TARGET ${the_module})
	set_target_properties(${the_module} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
	# For older GCC/libstdc++ (<9) we may need to link stdc++fs
	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
		target_link_libraries(${the_module} PRIVATE stdc++fs)
	endif()
endif()

# If you need to add special include directories, libraries or compile flags,
# add them below using the OpenCV contrib module helper macros, for example:
#
# ocv_include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
# ocv_module_compile_options(-Wall)
#
