cmake_minimum_required(VERSION 2.6)

project(mylib)

include_directories(include)

option (WITH_TESTS "build tests" ON)

if (WITH_TESTS)
  add_subdirectory(test)
endif()
