cmake_minimum_required(VERSION 2.6)
PROJECT(<%= library_name %> CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindPkgConfig)
<%= find_packages %>
SET(RBIND_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/types.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/operations.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/conversions.cc")

add_custom_command(OUTPUT ${RBIND_SRC}
       COMMAND ruby "${CMAKE_CURRENT_SOURCE_DIR}/../rbind.rb")

ADD_LIBRARY(<%= library_name %> SHARED ${RBIND_SRC})
TARGET_LINK_LIBRARIES(<%= library_name %> <%= libs %>)

set(ROOT_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/../..)
install(TARGETS <%= library_name %> LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../<%= ruby_path %>)

# global install
if(NOT GEM_INSTALL)
    FIND_PACKAGE(Ruby)
    IF(NOT RUBY_INCLUDE_PATH)
        MESSAGE(STATUS "Ruby library not found. Cannot install ruby extensions")
    ELSE(NOT RUBY_INCLUDE_PATH)
        STRING(REGEX REPLACE ".*lib(32|64)?/?" "lib/" RUBY_LIBRARY_INSTALL_DIR ${RUBY_RUBY_LIB_DIR})
        INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../lib/ DESTINATION ${RUBY_LIBRARY_INSTALL_DIR}
            FILES_MATCHING PATTERN "*.rb" PATTERN "*.rbind" PATTERN "*.so" PATTERN "*.dylib")
    ENDIF(NOT RUBY_INCLUDE_PATH)
ENDIF(NOT GEM_INSTALL)
