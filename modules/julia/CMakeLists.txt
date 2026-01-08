if(NOT HAVE_JULIA)
    message(STATUS "Julia not found. Not compiling Julia Bindings. ${HAVE_JULIA}")
    ocv_module_disable(julia)
elseif(NOT PYTHON_DEFAULT_EXECUTABLE)
   message(WARNING "Python required for Julia bindings...disabling")
   ocv_module_disable(julia)
endif()

# WARN_MIXED_PRECISION
macro(WARN_MIXED_PRECISION COMPILER_BITNESS JULIA_BITNESS)
    set(MSG "Your compiler is ${COMPILER_BITNESS}-bit")
    set(MSG "${MSG} but your version of Julia is ${JULIA_BITNESS}-bit.")
    set(MSG "${MSG} To build Julia bindings, please switch to a ${JULIA_BITNESS}-bit compiler.")
    message(WARNING ${MSG})
endmacro()


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

if(NOT JlCxx_DIR)
    execute_process(
        COMMAND "${Julia_EXECUTABLE}" --startup-file=no -e "using CxxWrap; print(CxxWrap.CxxWrapCore.prefix_path())"
        OUTPUT_VARIABLE JlCxx_DIR
    )
endif()

if(JlCxx_DIR)
    if(EXISTS ${JlCxx_DIR}/JlCxxConfig.cmake)
    else()
        message(STATUS "JlCxx found but not source build - disabling Julia module")
        ocv_module_disable(julia)
    endif()
endif()

find_package(JlCxx QUIET)

if(NOT JlCxx_FOUND)
    message(STATUS "JlCxx not found")
    ocv_module_disable(julia)
    return()
else()
    message(STATUS "JlCxx_DIR: ${JlCxx_DIR}")
endif()

set(JlCxx_DIR "${JlCxx_DIR}" CACHE STRING ADVANCED)
set(HAVE_JULIA "YES" CACHE STRING ADVANCED)


set(the_description "The Julia bindings")
ocv_add_module(
    julia
    BINDINGS
    OPTIONAL
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    opencv_videoio
    opencv_dnn
    opencv_features2d
    opencv_objdetect
    opencv_calib3d
)

set(HDR_PARSER_PATH ${CMAKE_SOURCE_DIR}/modules/python/src2/hdr_parser.py)

set(opencv_hdrs "")
foreach(m ${OPENCV_MODULES_BUILD})
  list(APPEND opencv_hdrs ${OPENCV_MODULE_${m}_HEADERS})
endforeach(m)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/gen/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(COPY ${HDR_PARSER_PATH} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/gen)

message(STATUS "Generating Julia Binding Files")

execute_process(
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gen"
                COMMAND ${PYTHON_DEFAULT_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/gen/gen_all.py" ${CMAKE_SOURCE_DIR}/modules ${OPENCV_MODULES_BUILD}
                )

file(COPY ${CMAKE_CURRENT_BINARY_DIR}/gen/cpp_files/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/gen/autogen_cpp)
file(COPY ${CMAKE_CURRENT_BINARY_DIR}/gen/jl_cxx_files/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/gen/autogen_jl)

ocv_glob_module_sources()
ocv_module_include_directories()
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wmissing-prototypes -Wmissing-declarations)

ocv_add_library(${the_module} SHARED ${OPENCV_MODULE_${the_module}_HEADERS}
                                     ${OPENCV_MODULE_${the_module}_SOURCES}
                                     "${CMAKE_CURRENT_BINARY_DIR}/gen/autogen_cpp/cv_core.cpp")

if(NOT JULIA_PKG_INSTALL_PATH)
    set(JULIA_PKG_INSTALL_PATH ${CMAKE_BINARY_DIR})
endif()
set(JULIA_PKG_INSTALL_PATH_HOOK ${JULIA_PKG_INSTALL_PATH} CACHE STRING "" FORCE)
mark_as_advanced(JULIA_PKG_INSTALL_PATH_HOOK)


message(STATUS "Installing ${the_module} bindings at ${JULIA_PKG_INSTALL_PATH}")


install(DIRECTORY package/ DESTINATION ${JULIA_PKG_INSTALL_PATH})
install(TARGETS ${the_module} LIBRARY DESTINATION ${JULIA_PKG_INSTALL_PATH}/OpenCV/src/lib)


if(JULIA_PKG_INSTALL_ENV)
    set(JULIA_PKG_EXECS "Pkg.activate(${JULIA_PKG_INSTALL_ENV});${JULIA_PKG_EXECS}")
endif()


set(JULIA_COMMAND "\"${Julia_EXECUTABLE}\" -e \"using Pkg; ${JULIA_PKG_EXECS}\"")
# message(STATUS "Installing Julia bindings using ${JULIA_COMMAND}")


if(NOT INSTALL_CREATE_DISTRIB)
    install(CODE "
    set(JULIA_PKG_PATH \"${JULIA_PKG_INSTALL_PATH}/OpenCV\")
    execute_process(COMMAND \"${Julia_EXECUTABLE}\" ${CMAKE_CURRENT_LIST_DIR}/package/install_package.jl \${JULIA_PKG_PATH} OUTPUT_VARIABLE JULIA_INSTALL_OUT)
    message(STATUS \"Install output: \${JULIA_INSTALL_OUT}\")
    ")
endif()


# ocv_create_module()

# ocv_add_accuracy_tests()
# ocv_add_perf_tests()
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

# targets# opencv_julia_sources --> opencv_julia


add_custom_command(TARGET ${the_module}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/package/OpenCV ${CMAKE_BINARY_DIR}/OpenCV
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/gen/autogen_jl/ ${CMAKE_BINARY_DIR}/OpenCV/src
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${the_module}> ${CMAKE_BINARY_DIR}/OpenCV/src/lib/libopencv_julia
                   COMMENT "Copying over julia package"
                   )



if (BUILD_TESTS)
    add_subdirectory(test)
endif()
