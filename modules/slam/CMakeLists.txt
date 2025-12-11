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

# Detect optional dependencies and expose preprocessor macros so sources
# can use #if defined(HAVE_SFM) / #if defined(HAVE_G2O)
if(TARGET ${the_module})
	# opencv_sfm is declared OPTIONAL above; if its target exists enable HAVE_SFM
	if(TARGET opencv_sfm)
		message(STATUS "slam: opencv_sfm target found — enabling HAVE_SFM")
		target_compile_definitions(${the_module} PRIVATE HAVE_SFM=1)
			# also add a global definition so header-parsers (python generator)
			# running during build/config time can evaluate #if HAVE_SFM
			add_definitions(-DHAVE_SFM=1)
		target_link_libraries(${the_module} PRIVATE opencv_sfm)
	else()
		message(STATUS "slam: opencv_sfm not found — HAVE_SFM disabled")
	endif()

	# Try to detect g2o (header + library). This is a best-effort detection: we
	# look for a common g2o header and try to find a library name. Users on
	# Windows should install/build g2o and make it discoverable to CMake.
	find_path(G2O_INCLUDE_DIR NAMES g2o/core/sparse_optimizer.h)
	find_library(G2O_LIBRARY NAMES g2o_core g2o)
	if(G2O_INCLUDE_DIR AND G2O_LIBRARY)
		message(STATUS "slam: g2o found (headers and library) — enabling HAVE_G2O")
		target_include_directories(${the_module} PRIVATE ${G2O_INCLUDE_DIR})
		target_link_libraries(${the_module} PRIVATE ${G2O_LIBRARY})
		target_compile_definitions(${the_module} PRIVATE HAVE_G2O=1)
			# also expose globally so external header parsers can evaluate #if HAVE_G2O
			add_definitions(-DHAVE_G2O=1)
	else()
		message(STATUS "slam: g2o not found — HAVE_G2O disabled")
	endif()
endif()

# Ensure C++17 is enabled for this module (required for <filesystem>)
if(TARGET ${the_module})
	set_target_properties(${the_module} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
	# For older GCC/libstdc++ (<9) we may need to link stdc++fs
	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
		target_link_libraries(${the_module} PRIVATE stdc++fs)
	endif()
endif()

# Raise Doxygen graph node limit to avoid include-graph generation failures
set(DOXYGEN_DOT_GRAPH_MAX_NODES 512 CACHE STRING "Max nodes for Doxygen dot graphs" FORCE)

# If you need to add special include directories, libraries or compile flags,
# add them below using the OpenCV contrib module helper macros, for example:
#
# ocv_include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
# ocv_module_compile_options(-Wall)
#
