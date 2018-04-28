set(the_description "OGRE 3D Visualiser.")

find_package(OGRE 1.10 QUIET)

if(NOT OGRE_FOUND)
  message(STATUS "Module opencv_ovis disabled because OGRE3D was not found")
  ocv_module_disable(ovis)
elseif(OGRE_VERSION VERSION_LESS 1.10 OR OGRE_VERSION VERSION_GREATER 2.0)
  message(STATUS "Module opencv_ovis disabled because of incompatible OGRE3D version (${OGRE_VERSION})")
  ocv_module_disable(ovis)
elseif(OGRE_VERSION VERSION_GREATER 1.10) # we need C++11 for OGRE 1.11
  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qstd=c++11")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif()
endif()

include_directories(${OGRE_INCLUDE_DIRS})
link_directories(${OGRE_LIBRARY_DIRS})

ocv_define_module(ovis opencv_core opencv_imgproc opencv_calib3d WRAP python)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-parameter)
ocv_target_link_libraries(${the_module} ${OGRE_LIBRARIES})
