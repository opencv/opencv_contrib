set(the_description "OGRE 3D Visualiser.")

find_package(OGRE QUIET CONFIG)

if(NOT OGRE_FOUND)
  message(STATUS "Module opencv_ovis disabled because OGRE3D was not found")
  ocv_module_disable(ovis)
elseif(OGRE_VERSION VERSION_LESS 1.11.5)
  message(STATUS "Module opencv_ovis disabled because of incompatible OGRE3D version (${OGRE_VERSION})")
  ocv_module_disable(ovis)
else() # we need C++11 for OGRE 1.11
  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qstd=c++11")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif()
endif()

include_directories(${OGRE_INCLUDE_DIRS})
link_directories(${OGRE_LIBRARY_DIRS})

ocv_add_module(ovis opencv_core opencv_imgproc opencv_calib3d WRAP python)
ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()

ocv_add_samples(opencv_aruco)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-parameter)
ocv_target_link_libraries(${the_module} ${OGRE_LIBRARIES})
