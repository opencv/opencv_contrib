set(the_description "Non-rigid 3D reconstruction")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DFORCE_INLINERS -D_MWAITXINTRIN_H_INCLUDED")

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)

ocv_define_module(dynamicfusion opencv_core opencv_viz opencv_imgproc)

ocv_warnings_disable(CMAKE_CXX_FLAGS
        -Wundef
        -Wshadow
        -Wsign-compare
        -Wmissing-declarations
        -Wunused-but-set-variable
        -Wunused-parameter
        -Wunused-function
        )

find_package(Ceres QUIET)

if (NOT Ceres_FOUND)
    message(STATUS "${the_module} has been disabled because Ceres library has not been found")
    ocv_module_disable(${the_module})
else()
    include_directories(${CERES_INCLUDE_DIRS})
    ocv_target_link_libraries(${the_module} ${CERES_LIBRARIES})
endif()
if(NOT HAVE_CUDA)
    message(STATUS "${the_module} currently requires CUDA to run")
    ocv_module_disable(${the_module})
else()
    include_directories(${CUDA_INCLUDE_DIRS})
    ocv_target_link_libraries(${the_module} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
endif()
