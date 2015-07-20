set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenNL)
if(OPENNL_FOUND)
    message(STATUS "OpenNL:   YES")
    set(HAVE_OPENNL 1)
else()
    message(STATUS "OpenNL:   NO")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rgbd_config.hpp.in
               ${CMAKE_BINARY_DIR}/rgbd_config.hpp @ONLY)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(${OPENNL_FOUND})
    include_directories(${OPENNL_INCLUDE_DIR})
endif()

set(the_description "RGBD algorithms")
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
ocv_define_module(rgbd opencv_core opencv_calib3d opencv_highgui opencv_imgproc opencv_structured_light WRAP python)

if(${OPENNL_FOUND})
    target_link_libraries(opencv_rgbd ${OPENNL_LIBS})
endif()
