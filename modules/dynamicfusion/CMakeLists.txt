set(the_description "Non-rigid 3D reconstruction")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DFORCE_INLINERS -D_MWAITXINTRIN_H_INCLUDED")

#
## ---[ Solution name
#
## ---[ utility
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")
include(cmake/Utils.cmake)
#include(cmake/Targets.cmake)
#
## ---[ find dependencies
find_package(CUDA REQUIRED)
#find_package(OpenNI)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${OPENNI_INCLUDE_DIR} "kfusion/src/utils" "kfusion/include/nanoflann")
#if(OPENNI_FOUND)
#    message("FOUND OPENNI AT: ${OPENNI_INCLUDE_DIR}")
#endif()
## ---[ misc settings
#if(USE_PROJECT_FOLDERS)
#  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
#  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
#endif()
#
## ---[ cuda settings
#set(HAVE_CUDA 1)
#list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_20,code=sm_20;-gencode;arch=compute_20,code=sm_21;-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_61,code=sm_61")
#
#if(UNIX OR APPLE)
#  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC;")
#endif()
#
#warnings_disable(CMAKE_CXX_FLAGS /wd4985)
#
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")
#add_module_library(kfusion)
#if(OPENNI_FOUND)
#    target_compile_definitions(kfusion PRIVATE OPENNI_FOUND=1)
#endif()
#target_link_libraries(kfusion ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${OPENNI_LIBRARY} ${CERES_LIBRARIES})
#add_subdirectory(apps)
#
#if(BUILD_TESTS)
#  find_package(GTest REQUIRED)
#  if( GTEST_FOUND )
#    message( "Found Gtest at ${GTEST_ROOT}")
#    message( "GTest Libs: ${GTEST_BOTH_LIBRARIES}")
#    message( "GTest Include: ${GTEST_INCLUDE_DIRS}")
#    include_directories(${GTEST_INCLUDE_DIRS})
#    add_subdirectory(tests)
#  endif()
#endif()
find_package(Boost REQUIRED COMPONENTS system filesystem program_options)
ocv_define_module(dynamicfusion opencv_core opencv_calib3d opencv_viz opencv_highgui)

ocv_warnings_disable(CMAKE_CXX_FLAGS
        -Wundef
        -Wshadow
        -Wsign-compare
        -Wmissing-declarations
        -Wunused-but-set-variable
        -Wunused-parameter
        -Wunused-function
        )
