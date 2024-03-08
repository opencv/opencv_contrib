function(ocv_add_tool the_target)
  cmake_parse_arguments(${the_target} "" "" "MODULES" ${ARGN})
  ocv_check_dependencies(${${the_target}_MODULES})
  if(NOT OCV_DEPENDENCIES_FOUND)
    return()
  endif()

  project(${the_target})

  ocv_include_directories(${the_target} PRIVATE "${OpenCV_SOURCE_DIR}/include/opencv")
  ocv_include_modules_recurse(${the_target} ${${the_target}_MODULES})

  file(GLOB ${the_target}_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

  add_executable(${the_target} ${${the_target}_SOURCES})

  ocv_target_link_libraries(${the_target} ${${the_target}_MODULES})

  set_target_properties(${the_target} PROPERTIES
                          DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                          ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                          RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                          OUTPUT_NAME ${the_target})

  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${the_target} PROPERTIES FOLDER "applications")
  endif()

  install(TARGETS ${the_target} OPTIONAL RUNTIME DESTINATION bin COMPONENT main)
endfunction()


add_subdirectory(waldboost_detector)
# add_subdirectory(traincascade)
# add_subdirectory(createsamples)
add_subdirectory(annotation)
add_subdirectory(visualisation)