cmake_minimum_required (VERSION 3.0)
project (myproject)


set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

if(WIN32)
    set( OpenCV_DIR "C:\\OpenCV\\build" )
elseif(UNIX)
    set( OpenCV_DIR "/usr/share/OpenCV" )
    set( PKG_CONFIG_PATH "/usr/lib/pkgconfig" )
endif()

find_package(OpenCV)
if( OpenCV_FOUND )
    message( "Found OpenCV" )
    message( "Includes: " ${OpenCV_INCLUDE_DIRS} )
else()
    message( "OpenCV not found, woot" )
endif( OpenCV_FOUND )

add_executable (alpha src/alphac.cpp)
target_link_libraries( alpha ${OpenCV_LIBS} )
