set(the_description "AprilTag 3 fiducial detection")

find_package(Threads QUIET)

if(Threads_FOUND)
    if(CMAKE_USE_PTHREADS_INIT)
        set(HAVE_PTHREAD 1)
    endif()
else()
    set (NO_THREADS 1)
endif()

set(APRILTAG_INCLUDES
    "${CMAKE_CURRENT_LIST_DIR}/src/apriltag"
    "${OpenCV_SOURCE_DIR}/include/opencv"
)

if(HAVE_PTHREAD)
    set(PTHREADS_LIBS
        Threads::Threads
    )
elseif(NOT HAVE_PTHREAD AND MSVC)
    list(APPEND APRILTAG_INCLUDES "${CMAKE_CURRENT_LIST_DIR}/src/pthreads4w")
    list(APPEND APRILTAG_INCLUDES "${CMAKE_CURRENT_LIST_DIR}/src/pthreads4w/pthreads4w-src")
    set(PTHREADS_LIBS
        pthreads4w
    )
else()
    set(DISABLE_MSG "Module opencv_apriltag disabled because pthreads dependency is not found")
    message(STATUS ${DISABLE_MSG})
    ocv_module_disable(apriltag)
endif()

set(APRILTAG_LIBS
    apriltag
)

ocv_add_module(apriltag
    opencv_core
    opencv_calib3d
    opencv_imgproc
    WRAP python java
)

if(UNIX)
    if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()
endif()

ocv_include_directories( ${APRILTAG_INCLUDES} )
ocv_module_include_directories()

# source files
FILE(GLOB OPENCV_APRILTAG_SRC src/*.cpp)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB OPENCV_APRILTAG_HDRS include/opencv2/apriltag.hpp include/opencv2/apriltag/*.hpp)

ocv_set_module_sources(HEADERS ${OPENCV_APRILTAG_HDRS}
                       SOURCES ${OPENCV_APRILTAG_SRC})

ocv_create_module()

if(NOT CMAKE_VERSION VERSION_LESS 2.8.11) # See ocv_target_include_directories() implementation
    if(TARGET ${the_module})
        get_target_property(__include_dirs ${the_module} INCLUDE_DIRECTORIES)
        include_directories(${__include_dirs})
    endif()
endif()

include_directories(${OCV_TARGET_INCLUDE_DIRS_${the_module}})
if(NOT HAVE_PTHREAD AND MSVC)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/src/pthreads4w" "${CMAKE_CURRENT_BINARY_DIR}/src/pthreads4w")
endif()
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/src/apriltag" "${CMAKE_CURRENT_BINARY_DIR}/src/apriltag")

if(NOT HAVE_PTHREAD AND MSVC)
    ocv_target_link_libraries(${the_module} winmm ${PTHREADS_LIBS} ${APRILTAG_LIBS})
else()
    ocv_target_link_libraries(${the_module} ${PTHREADS_LIBS} ${APRILTAG_LIBS})
endif()

### CREATE OPENCV APRILTAG TESTS ###

ocv_add_accuracy_tests()
