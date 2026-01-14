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
	# Build-only shim for 3rd-party libs that include legacy <opencv2/core/core.hpp>
	# (DBoW3 does this). The real header errors out during OpenCV's own build.
	target_include_directories(${the_module} BEFORE PRIVATE ${CMAKE_CURRENT_LIST_DIR}/shim)

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

	# Optional DBoW3 for BoW-based loop detection
	find_package(DBoW3 QUIET)
	if(DBoW3_FOUND)
		set(_dbow3_ok FALSE)
		if(TARGET DBoW3::DBoW3)
			target_link_libraries(${the_module} PRIVATE DBoW3::DBoW3)
			set(_dbow3_ok TRUE)
		else()
			# Some DBoW3 installs do not export an imported target. In that case do
			# NOT try to synthesize an imported target via IMPORTED_LOCATION because
			# DBoW3_LIBRARIES is often a LIST (e.g. OpenCV targets + DBoW3), which
			# would generate invalid Makefile rules.
			if(DBoW3_INCLUDE_DIRS)
				target_include_directories(${the_module} PRIVATE ${DBoW3_INCLUDE_DIRS})
			elseif(DBoW3_INCLUDE_DIR)
				target_include_directories(${the_module} PRIVATE ${DBoW3_INCLUDE_DIR})
			endif()
			# Link only the DBoW3 library itself. Some legacy FindDBoW3 scripts set
			# DBoW3_LIBRARIES to include OpenCV libraries (or even opencv_slam), which
			# would create circular/self links in an OpenCV superbuild.
			set(_dbow3_libs "")
			if(DBoW3_LIBRARIES)
				set(_dbow3_libs ${DBoW3_LIBRARIES})
			elseif(DBoW3_LIBRARY)
				set(_dbow3_libs ${DBoW3_LIBRARY})
			endif()
			if(_dbow3_libs)
				# Prefer entries that look like the DBoW3 library.
				set(_dbow3_only ${_dbow3_libs})
				list(FILTER _dbow3_only INCLUDE REGEX ".*[Dd][Bb][Oo][Ww]3.*")
				if(_dbow3_only)
					set(_dbow3_libs ${_dbow3_only})
				else()
					# Otherwise, remove OpenCV libs/targets to avoid circular dependencies.
					list(FILTER _dbow3_libs EXCLUDE REGEX "^(opencv_.*|OpenCV::.*)$")
					list(FILTER _dbow3_libs EXCLUDE REGEX "^opencv_slam$")
				endif()
				target_link_libraries(${the_module} PRIVATE ${_dbow3_libs})
				set(_dbow3_ok TRUE)
			endif()
		endif()
		if(_dbow3_ok)
			message(STATUS "slam: DBoW3 found — enabling HAVE_DBOW3")
			target_compile_definitions(${the_module} PRIVATE HAVE_DBOW3=1)
			add_definitions(-DHAVE_DBOW3=1)
		else()
			message(WARNING "slam: DBoW3 found but no link information (DBoW3::DBoW3 / DBoW3_LIBRARIES / DBoW3_LIBRARY). BoW disabled.")
		endif()
	else()
		message(STATUS "slam: DBoW3 not found — HAVE_DBOW3 disabled")
	endif()

	# Try to detect g2o (headers + required libs). We need core + stuff +
	# types_sba for the SE3/pose graph edges used in this module. Optional libs
	# are linked when available (slam3d/sim3/solvers) to improve compatibility
	# with different g2o builds.
	find_path(G2O_INCLUDE_DIR NAMES g2o/core/sparse_optimizer.h)
	find_library(G2O_CORE_LIBRARY NAMES g2o_core g2o)
	find_library(G2O_STUFF_LIBRARY NAMES g2o_stuff)
	find_library(G2O_TYPES_SBA_LIBRARY NAMES g2o_types_sba)

	set(_g2o_required_libs ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_TYPES_SBA_LIBRARY})
	list(FILTER _g2o_required_libs EXCLUDE REGEX "NOTFOUND")

	if(G2O_INCLUDE_DIR AND G2O_CORE_LIBRARY AND G2O_STUFF_LIBRARY AND G2O_TYPES_SBA_LIBRARY)
		message(STATUS "slam: g2o found (core/stuff/types_sba) — enabling HAVE_G2O")
		target_include_directories(${the_module} PRIVATE ${G2O_INCLUDE_DIR})

		set(_g2o_libs ${_g2o_required_libs})
		foreach(_g2o_candidate g2o_types_slam3d g2o_types_slam3d_addons g2o_types_sim3 g2o_types_data g2o_solver_csparse g2o_solver_cholmod g2o_solver_eigen g2o_solver_dense g2o_solver_pcg)
			find_library(_g2o_lib NAMES ${_g2o_candidate})
			if(_g2o_lib)
				list(APPEND _g2o_libs ${_g2o_lib})
			endif()
		endforeach()
		list(REMOVE_DUPLICATES _g2o_libs)
		target_link_libraries(${the_module} PRIVATE ${_g2o_libs})
		target_compile_definitions(${the_module} PRIVATE HAVE_G2O=1)
			# also expose globally so external header parsers can evaluate #if HAVE_G2O
			add_definitions(-DHAVE_G2O=1)
	else()
		message(STATUS "slam: g2o not found — HAVE_G2O disabled (need headers + core/stuff/types_sba)")
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

# If you need to add special include directories, libraries or compile flags,
# add them below using the OpenCV contrib module helper macros, for example:
#
# ocv_include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
# ocv_module_compile_options(-Wall)
#
