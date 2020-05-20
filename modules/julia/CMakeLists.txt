include(${CMAKE_CURRENT_LIST_DIR}/FindJulia.cmake)

if(NOT Julia_FOUND)
    ocv_module_disable(julia)
elseif(NOT PYTHON3_EXECUTABLE)
#    message(WARNING "Python 3 required for Julia bindings...disabling")
#    ocv_module_disable(julia)
endif()

math(EXPR ARCH "${CMAKE_SIZEOF_VOID_P} * 8")
if (${ARCH} EQUAL 32 AND ${Julia_WORD_SIZE} MATCHES "64")
    warn_mixed_precision("32" "64")
    ocv_module_disable(julia)
    return()
elseif (${ARCH} EQUAL 64 AND NOT ${Julia_WORD_SIZE} MATCHES "64")
    warn_mixed_precision("64" "32")
    ocv_module_disable(julia)
    return()
endif()

execute_process(
    COMMAND "${Julia_EXECUTABLE}" --startup-file=no -e "using CxxWrap; print(CxxWrap.CxxWrapCore.prefix_path())"
    OUTPUT_VARIABLE JlCxx_DIR
)


find_package(JlCxx REQUIRED)

if(NOT JlCxx_FOUND)
    ocv_module_disable(julia)
    return()
else()
    message(STATUS "JlCxx_DIR: ${JlCxx_DIR}")
endif()

set(the_description "The Julia bindings")
ocv_add_module(julia    BINDINGS
                                 opencv_core
                                 opencv_imgproc 
                                 opencv_imgcodecs
                                 opencv_objdetect
                                 opencv_highgui
                                 opencv_features2d
                                 opencv_videoio
)

ocv_glob_module_sources()
ocv_module_include_directories()

ocv_add_library(${the_module} SHARED ${OPENCV_MODULE_${the_module}_HEADERS} ${OPENCV_MODULE_${the_module}_SOURCES})

if(NOT JULIA_PKG_INSTALL_PATH)
    set(JULIA_PKG_INSTALL_PATH ${CMAKE_CURRENT_BINARY_DIR})
endif()
message(STATUS "Installing ${the_module} bindings at ${JULIA_PKG_INSTALL_PATH}")


install(DIRECTORY package/ DESTINATION ${JULIA_PKG_INSTALL_PATH})
install(TARGETS ${the_module} LIBRARY DESTINATION ${JULIA_PKG_INSTALL_PATH}/OpenCV/src/lib)


if(JULIA_PKG_INSTALL_ENV)
    set(JULIA_PKG_EXECS "Pkg.activate(${JULIA_PKG_INSTALL_ENV});${JULIA_PKG_EXECS}")
endif()


set(JULIA_COMMAND "\"${Julia_EXECUTABLE}\" -e \"using Pkg; ${JULIA_PKG_EXECS}\"")
message(STATUS "Installing Julia bindings using ${JULIA_COMMAND}")


install(CODE "
set(JULIA_PKG_PATH \"${JULIA_PKG_INSTALL_PATH}/OpenCV\")
execute_process(COMMAND \"${Julia_EXECUTABLE}\" ${CMAKE_CURRENT_LIST_DIR}/package/install_package.jl \${JULIA_PKG_PATH} OUTPUT_VARIABLE JULIA_INSTALL_OUT)


# set(JULIA_PKG_EXECS \"Pkg.develop(PackageSpec(path=\\\\\\\"${JULIA_PKG_INSTALL_PATH}/OpenCV\\\\\\\"));\")

# if(${JULIA_PKG_INSTALL_ENV})
#     set(JULIA_PKG_EXECS \"Pkg.activate(${JULIA_PKG_INSTALL_ENV});\${JULIA_PKG_EXECS}\")
# endif()
# set(JULIA_COMMAND \"\\\"using Pkg; \${JULIA_PKG_EXECS}\\\"\")

# message(STATUS \"Installing Julia bindings using \${JULIA_COMMAND}\")

# execute_process(COMMAND \"${Julia_EXECUTABLE}\" -e \${JULIA_COMMAND} OUTPUT_VARIABLE JULIA_INSTALL_OUT)

message(STATUS \"Install output: \${JULIA_INSTALL_OUT}\")
")

# set(JULIA_COMMAND \"\"${Julia_EXECUTABLE}\" -e \"using Pkg; \${JULIA_PKG_EXECS}\"\")

# ocv_create_module()

ocv_add_accuracy_tests()
ocv_add_perf_tests()
ocv_add_samples()

message(STATUS     ${OPENCV_MODULE_${the_module}_DEPS_TO_LINK})

ocv_target_link_libraries(${the_module} PUBLIC    ${OPENCV_MODULE_${the_module}_DEPS_TO_LINK}
                                        INTERFACE ${OPENCV_MODULE_${the_module}_DEPS_TO_LINK}
)
ocv_target_link_libraries(${the_module} PUBLIC    ${OPENCV_MODULE_${the_module}_DEPS_EXT}
                                        INTERFACE ${OPENCV_MODULE_${the_module}_DEPS_EXT}
)
ocv_target_link_libraries(${the_module} PRIVATE ${OPENCV_LINKER_LIBS} ${OPENCV_HAL_LINKER_LIBS} ${IPP_LIBS} ${ARGN})

ocv_target_link_libraries(${the_module} JlCxx::cxxwrap_julia)
ocv_target_link_libraries(${the_module} JlCxx::cxxwrap_julia_stl)


if (BUILD_TESTS)
    add_subdirectory(test)
endif()

