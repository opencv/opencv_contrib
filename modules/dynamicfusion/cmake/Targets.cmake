################################################################################################
# short command to setup source group
function(kf_source_group group)
  cmake_parse_arguments(VW_SOURCE_GROUP "" "" "GLOB" ${ARGN})
  file(GLOB srcs ${VW_SOURCE_GROUP_GLOB})
  #list(LENGTH ${srcs} ___size)
  #if (___size GREATER 0)
    source_group(${group} FILES ${srcs})
  #endif()
endfunction()


################################################################################################
# short command getting sources from standard directores
macro(pickup_std_sources)
  kf_source_group("Include" GLOB "include/${module_name}/*.h*")
  kf_source_group("Include\\cuda" GLOB "include/${module_name}/cuda/*.h*")
  kf_source_group("Source" GLOB "src/*.cpp" "src/*.h*")
  kf_source_group("Source\\utils" GLOB "src/utils/*.cpp" "src/utils/*.h*")
  kf_source_group("Source\\cuda" GLOB "src/cuda/*.c*" "src/cuda/*.h*")
  FILE(GLOB_RECURSE sources include/${module_name}/*.h* src/*.cpp src/*.h* src/cuda/*.h* src/cuda/*.c*)
endmacro()


################################################################################################
# short command for declaring includes from other modules
macro(declare_deps_includes)
  foreach(__arg ${ARGN})
    get_filename_component(___path "${CMAKE_SOURCE_DIR}/modules/${__arg}/include" ABSOLUTE)
    if (EXISTS ${___path})
      include_directories(${___path})
    endif()
  endforeach()

  unset(___path)
  unset(__arg)
endmacro()


################################################################################################
# short command for setting defeault target properties
function(default_properties target)
  set_target_properties(${target} PROPERTIES
    DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    if (NOT ${target} MATCHES "^test_")
      install(TARGETS ${the_target} RUNTIME DESTINATION ".")
    endif()
endfunction()

function(test_props target)
  #os_project_label(${target} "[test]")
  if(USE_PROJECT_FOLDERS)
    set_target_properties(${target} PROPERTIES FOLDER "Tests")
  endif()
endfunction()

function(app_props target)
  #os_project_label(${target} "[app]")
  if(USE_PROJECT_FOLDERS)
    set_target_properties(${target} PROPERTIES FOLDER "Apps")
  endif()
endfunction()


################################################################################################
# short command for setting defeault target properties
function(default_properties target)
  set_target_properties(${target} PROPERTIES
    DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    if (NOT ${target} MATCHES "^test_")
      install(TARGETS ${the_target} RUNTIME DESTINATION ".")
    endif()
endfunction()


################################################################################################
# short command for adding library module
macro(add_module_library name)
  set(module_name ${name})
  pickup_std_sources()
  include_directories(include src src/cuda src/utils)

  set(__has_cuda OFF)
  check_cuda(__has_cuda)

  set(__lib_type STATIC)
  if (${ARGV1} MATCHES "SHARED|STATIC")
    set(__lib_type ${ARGV1})
  endif()

  if (__has_cuda)
    cuda_add_library(${module_name} ${__lib_type} ${sources})
  else()
    add_library(${module_name} ${__lib_type} ${sources})
  endif()

  if(MSVC)
    set_target_properties(${module_name} PROPERTIES DEFINE_SYMBOL KFUSION_API_EXPORTS)
  else()
    add_definitions(-DKFUSION_API_EXPORTS)
  endif()

  default_properties(${module_name})

  if(USE_PROJECT_FOLDERS)
    set_target_properties(${module_name} PROPERTIES FOLDER "Libraries")
  endif()

  set_target_properties(${module_name} PROPERTIES INSTALL_NAME_DIR lib)

  install(TARGETS ${module_name}
    RUNTIME DESTINATION bin COMPONENT main
    LIBRARY DESTINATION lib COMPONENT main
    ARCHIVE DESTINATION lib COMPONENT main)

  install(DIRECTORY include/ DESTINATION include/ FILES_MATCHING PATTERN "*.h*")
endmacro()

################################################################################################
# short command for adding application module
macro(add_application target sources)
  add_executable(${target} ${sources})
  default_properties(${target})
  app_props(${target})
endmacro()
