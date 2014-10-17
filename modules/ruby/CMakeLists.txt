# ----------------------------------------------------------------------------
#  CMake file for ruby support
# ----------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake:${CMAKE_MODULE_PATH}")
FIND_PACKAGE(Ruby)
if(IOS OR ANDROID OR BUILD_opencv_world OR NOT RUBY_RUBY_LIB_DIR)
    ocv_module_disable(ruby)
endif()

set(MODULE_NAME ruby)
set(the_description "The ruby bindings")
set(MODULE_INSTALL_SUBDIR ruby)

# get other modules
set(ruby_candidate_deps "")
foreach(mp ${OPENCV_MODULES_PATH} ${OPENCV_EXTRA_MODULES_PATH})
    file(GLOB names "${mp}/*")
    foreach(m IN LISTS names)
        if(IS_DIRECTORY ${m})
            get_filename_component(m ${m} NAME)
            list(APPEND ruby_candidate_deps "opencv_${m}")
        endif()
    endforeach(m)
endforeach(mp)

# module blacklist
ocv_list_filterout(ruby_candidate_deps "^opencv_matlab$")
ocv_list_filterout(ruby_candidate_deps "^opencv_python2$")
ocv_list_filterout(ruby_candidate_deps "^opencv_python3$")
ocv_list_filterout(ruby_candidate_deps "^opencv_java$")
ocv_list_filterout(ruby_candidate_deps "^opencv_cud(a|ev)")

# module blacklist due to python binding
# can be removed if python binding is blacklisting ruby
ocv_list_filterout(ruby_candidate_deps "^opencv_ts$")
ocv_list_filterout(ruby_candidate_deps "^opencv_adas$")
ocv_list_filterout(ruby_candidate_deps "^opencv_tracking$")
ocv_list_filterout(ruby_candidate_deps "^opencv_bioinspired$")

# set dependencies
ocv_add_module(${MODULE_NAME} BINDINGS OPTIONAL ${ruby_candidate_deps})
ocv_module_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

# get headers to parse
set(opencv_ruby_hdrs "")
foreach(m IN LISTS OPENCV_MODULE_${the_module}_DEPS)
    list(APPEND opencv_ruby_hdrs ${OPENCV_MODULE_${m}_HEADERS})
endforeach(m)

# header blacklist
ocv_list_filterout(opencv_ruby_hdrs ".h$")
ocv_list_filterout(opencv_ruby_hdrs "cuda")
ocv_list_filterout(opencv_ruby_hdrs "cudev")

# setup parser command
SET(RBIND_HDR src/types.h src/conversions.hpp)
SET(RBIND_SRC src/types.cc src/operations.cc src/conversions.cc)
add_custom_command(OUTPUT ${RBIND_SRC}
    COMMAND ruby "${CMAKE_CURRENT_SOURCE_DIR}/parser/parse.rb" "${CMAKE_CURRENT_BINARY_DIR}/src" ${opencv_ruby_hdrs}
    DEPENDS ${opencv_ruby_hdrs})

# setup target
ocv_add_library(${the_module} SHARED ${RBIND_SRC})
ocv_target_link_libraries(${the_module} ${OPENCV_MODULE_${the_module}_DEPS})

if(NOT INSTALL_CREATE_DISTRIB)
  # install library
  # install(TARGETS ${the_module}
  #     LIBRARY DESTINATION lib COMPONENT ruby)

  # install ruby files + library
  IF(RUBY_RUBY_LIB_DIR)
      STRING(REGEX REPLACE ".*lib(32|64)?/?" "lib/" RUBY_LIBRARY_INSTALL_DIR ${RUBY_RUBY_LIB_DIR})
      INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/ DESTINATION ${RUBY_LIBRARY_INSTALL_DIR}
          FILES_MATCHING PATTERN "*.rb" PATTERN "*.rbind")
      install(TARGETS ${the_module} LIBRARY DESTINATION "${RUBY_LIBRARY_INSTALL_DIR}/opencv/")
  ENDIF(RUBY_RUBY_LIB_DIR)
else()
endif()

# cleanup
unset(MODULE_NAME)
unset(MODULE_INSTALL_SUBDIR)
