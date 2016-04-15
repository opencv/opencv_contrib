if(NOT HAVE_QT5)
  ocv_module_disable(cvv)
  return()
endif()

set(the_description "Debug visualization framework")
ocv_define_module(cvv opencv_core opencv_imgproc opencv_features2d ${CVV_LIBRARIES} WRAP python)

# we need C++11 and want warnings:
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qstd=c++11 /W4")
  add_definitions(/D__func__=__FUNCTION__)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic")
endif()
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshadow -Wmissing-declarations)

# Qt5
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
foreach(dt5_dep Core Gui Widgets)
  add_definitions(${Qt5${dt5_dep}_DEFINITIONS})
  include_directories(${Qt5${dt5_dep}_INCLUDE_DIRS})
  list(APPEND CVV_LIBRARIES ${Qt5${dt5_dep}_LIBRARIES})
endforeach()
