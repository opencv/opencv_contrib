if(NOT HAVE_QT OR NOT HAVE_CXX11 OR QT_VERSION_MAJOR LESS 5)
  ocv_module_disable(cvv)
  return()
endif()

set(the_description "Debug visualization framework")
ocv_add_module(cvv opencv_core opencv_imgproc opencv_features2d WRAP python)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshadow -Wmissing-declarations)

# Qt
set(CVV_QT_MODULES Core Gui Widgets)
if(QT_VERSION_MAJOR EQUAL 6)
  list(APPEND CVV_QT_MODULES Core5Compat)
endif()

find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS ${CVV_QT_MODULES})

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
foreach(module ${CVV_QT_MODULES})
  list(APPEND CVV_LIBRARIES ${Qt${QT_VERSION_MAJOR}${module}_LIBRARIES})
endforeach()

ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module(${CVV_LIBRARIES})
ocv_add_accuracy_tests()
ocv_add_perf_tests()
ocv_add_samples()
