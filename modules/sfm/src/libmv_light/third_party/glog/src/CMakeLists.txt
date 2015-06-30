IF (WIN32)
  INCLUDE_DIRECTORIES(
    windows
    ../../msinttypes
  )
ELSE (WIN32)
  INCLUDE_DIRECTORIES(
    .
  )
ENDIF (WIN32)

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "-fPIC")
  #  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused -Wno-type-limits -Wno-empty-body")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

IF (WIN32)
  ADD_LIBRARY(glog
              logging.cc
              windows/port.cc
              raw_logging.cc
              utilities.cc
              vlog_is_on.cc)
ELSE (WIN32)
  ADD_LIBRARY(glog
              demangle.cc
              logging.cc
              raw_logging.cc
              signalhandler.cc
              symbolize.cc
              utilities.cc
              vlog_is_on.cc)
  TARGET_LINK_LIBRARIES(glog gflags)
ENDIF (WIN32)

MACRO (GLOG_TEST NAME)
  ADD_EXECUTABLE(${NAME} ${NAME}.cc)
  TARGET_LINK_LIBRARIES(${NAME} glog gflags pthread googletest)
  SET_TARGET_PROPERTIES(${NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${LIBMV_TESTS_OUTPUT_DIR})
  ADD_TEST(${NAME} ${LIBMV_TESTS_OUTPUT_DIR}/${NAME})
ENDMACRO (GLOG_TEST)

IF (NOT WIN32)
#GLOG_TEST(demangle_unittest)
# These require extra magic...
#GLOG_TEST(logging_striptest10)
#GLOG_TEST(logging_striptest2)
# I don't think logging_unittest can really be called a unittest, because it
# benchmarks the logging. Commented out to keep tests fast.
#GLOG_TEST(logging_unittest)
#GLOG_TEST(signalhandler_unittest)
#GLOG_TEST(stacktrace_unittest)
#GLOG_TEST(stl_logging_unittest)
#GLOG_TEST(symbolize_unittest)
#GLOG_TEST(utilities_unittest)

ENDIF (NOT WIN32)

LIBMV_INSTALL_THIRD_PARTY_LIB(glog)
