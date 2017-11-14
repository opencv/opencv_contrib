set(the_description "OGRE 3D Visualiser.")

find_package(OGRE 1.10 QUIET)

if(NOT OGRE_FOUND)
  message(STATUS "Module opencv_ovis disabled because OGRE3D was not found")
  ocv_module_disable(ovis)
elseif(OGRE_VERSION VERSION_LESS 1.10)
  message(STATUS "Module opencv_ovis disabled because OGRE3D version is less than 1.10 (${OGRE_VERSION})")
  ocv_module_disable(ovis)
endif()

include_directories(${OGRE_INCLUDE_DIRS}})
link_directories(${OGRE_LIBRARY_DIRS})

ocv_define_module(ovis opencv_core opencv_imgproc opencv_calib3d WRAP python)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-parameter)
ocv_target_link_libraries(${the_module} ${OGRE_LIBRARIES})
